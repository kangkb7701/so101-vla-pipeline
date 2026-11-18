"""
SO-101 Octo 파인튜닝 스크립트 (Pickup 50 에피소드 전용 단일 태스크 버전)
====================================================================
단일 태스크(Pick up) 학습을 통해 모델의 시각 인지 능력(과적합 여부)을 
1차적으로 검증하기 위해 하이퍼파라미터를 극도로 축소한 버전입니다.

[주요 변경점]
  - Dataset: so101_pickup 단일 로드 (50 episodes)
  - Train Steps: 3000 -> 2000 (데이터 감소에 따른 과적합 방지)
  - Warmup Steps: 200 -> 100
  - [NEW] Action Head Only FT: 학습률 3e-4 상향, Backbone 완전 동결
"""
from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm
import wandb

from octo.data.dataset import make_interleaved_dataset
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
# [수정] Backbone을 얼리면 VRAM 소모가 줄어들므로 Batch Size를 16으로 상향 (필요시 CLI로 조절 가능)
flags.DEFINE_integer("batch_size", 16, "Batch size for finetuning (16 recommended for Head-only on 24GB VRAM).")
flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)

# ===================================================================
# SO-101 전용 설정 (단일 태스크)
# ===================================================================
SO101_ACTION_DIM = 7        
SO101_ACTION_HORIZON = 4    
SO101_DATASET_PICKUP = "so101_pickup"       
# put_inside 데이터셋 제거됨
# ===================================================================


def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")

    wandb.init(name="finetune_so101_pickup_only_freezing", project="octo")

    # ---------------------------------------------------------------
    # 1. 사전학습 모델 로드
    # ---------------------------------------------------------------
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # ---------------------------------------------------------------
    # 2. 파인튜닝 데이터셋 로드 (Pickup 전용으로 수정)
    # ---------------------------------------------------------------
    logging.info("Loading finetuning dataset...")

    _dataset_kwargs = dict(
        image_obs_keys={
            "primary": "image_primary",   
            "wrist": "image_wrist",       
        },
        proprio_obs_key="state",
        language_key="language_instruction",
    )

    dataset = make_interleaved_dataset(
        dataset_kwargs_list=[
            # 오직 pickup 데이터셋 하나만 로드
            dict(name=SO101_DATASET_PICKUP, data_dir=FLAGS.data_dir, **_dataset_kwargs),
        ],
        # 단일 데이터셋이므로 sample_weights 제거
        shuffle_buffer_size=10000,              
        traj_transform_kwargs=dict(
            window_size=2,                      
            action_horizon=SO101_ACTION_HORIZON,
        ),
        frame_transform_kwargs=dict(
            resize_size={
                "primary": (256, 256),
                "wrist":   (128, 128),
            },
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .batch(FLAGS.batch_size)
        .iterator()
    )

    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    # ---------------------------------------------------------------
    # 3. 모델 설정 수정
    # ---------------------------------------------------------------
    logging.info("Updating model for SO-101 observation space...")
    config = pretrained_model.config

    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-180.0,
        high=180.0,
        obs_keys=["proprio"],
    )

    # ---------------------------------------------------------------
    # 4. 모델 초기화 + 사전학습 가중치 병합
    # ---------------------------------------------------------------
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics,
    )
    
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    # ---------------------------------------------------------------
    # 5. 옵티마이저 & 학습 상태 설정
    # ---------------------------------------------------------------
    # [수정] Head-only 학습을 위해 학습률을 1e-5에서 3e-4로 대폭 상향
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-4, 100), optax.constant_schedule(3e-4)], [100]
    )
    tx = optax.adamw(learning_rate)
    
    frozen_keys = list(model.config.get("optimizer", {}).get("frozen_keys", []))
    
    if FLAGS.freeze_transformer:
        # [수정] octo_transformer 이하의 모든 가중치(Attention, MLP 등)를 동결
        frozen_keys.append("octo_transformer.*")
        logging.info("❄️ Transformer backbone completely frozen. Training Action Head ONLY.")
        
    tx = freeze_weights(tx, model.params, frozen_keys)
    
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # ---------------------------------------------------------------
    # 6. 학습 루프
    # ---------------------------------------------------------------
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    NUM_TRAIN_STEPS = 2000
    LOG_INTERVAL = 50
    SAVE_INTERVAL = 500

    logging.info("Starting finetuning...")
    for i in tqdm.tqdm(range(NUM_TRAIN_STEPS), total=NUM_TRAIN_STEPS, dynamic_ncols=True):
        batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)
        
        if (i + 1) % LOG_INTERVAL == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
        
        if (i + 1) % SAVE_INTERVAL == 0:
            train_state.model.save_pretrained(
                step=i, checkpoint_path=FLAGS.save_dir
            )
    
    train_state.model.save_pretrained(
        step=NUM_TRAIN_STEPS - 1, checkpoint_path=FLAGS.save_dir
    )
    logging.info("Finetuning complete! Ready for evaluation.")

if __name__ == "__main__":
    app.run(main)
