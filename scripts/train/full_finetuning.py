"""
SO-101 Octo 파인튜닝 스크립트 (Full Fine-tuning & Diffusion 보존 버전)
====================================================================
so101_dataset_builder.py로 변환한 RLDS 데이터셋을 사용하여
Octo를 SO-101 로봇에 맞게 파인튜닝합니다.

[주요 아키텍처 변경점]
  - 트랜스포머 동결 해제 (freeze_transformer=False 기본값): 시각적 도메인 갭 극복
  - L1ActionHead 교체 로직 제거: 순정 Diffusion 헤드의 미세 제어 능력 및 사전학습 가중치 활용
  - Learning Rate 하향 (1e-5) 및 Warmup 축소 (200 step): 재앙적 망각 방지
  - Batch Size 하향 (8) 및 Train Step 대폭 축소 (3000): VRAM 관리 및 과적합 방지

사용법:
  python finetune_so101.py \
      --pretrained_path=hf://rail-berkeley/octo-small-1.5 \
      --data_dir=<tfds 빌드된 데이터셋 경로> \
      --save_dir=<체크포인트 저장 경로>
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
# [수정 1] Full Fine-tuning 시 VRAM OOM을 방지하기 위해 기본 배치 사이즈를 8로 하향
flags.DEFINE_integer("batch_size", 8, "Batch size for finetuning (8 recommended for 24GB VRAM).")
# [수정 2] 시각적 도메인 갭(SO-101 카메라 구도) 학습을 위해 기본적으로 트랜스포머 봉인 해제
flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)


# ===================================================================
# SO-101 전용 설정
# ===================================================================
SO101_ACTION_DIM = 7        # FK delta: [dx, dy, dz, drx, dry, drz, d_gripper]
SO101_ACTION_HORIZON = 4    # Octo 기본 action chunk 길이
SO101_DATASET_PICKUP     = "so101_pickup"       
SO101_DATASET_PUT_INSIDE = "so101_put_inside"   
# ===================================================================


def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # TensorFlow가 GPU 메모리를 점유하지 않도록 설정
    tf.config.set_visible_devices([], "GPU")

    # wandb 로깅 설정
    wandb.init(name="finetune_so101_full_diffusion", project="octo")

    # ---------------------------------------------------------------
    # 1. 사전학습 모델 로드
    # ---------------------------------------------------------------
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # ---------------------------------------------------------------
    # 2. 파인튜닝 데이터셋 로드
    # ---------------------------------------------------------------
    logging.info("Loading finetuning dataset...")

    _dataset_kwargs = dict(
        image_obs_keys={
            "primary": "image_primary",   # top 카메라 → Octo primary
            "wrist": "image_wrist",       # front 카메라 → Octo wrist
        },
        proprio_obs_key="state",
        language_key="language_instruction",
    )

    dataset = make_interleaved_dataset(
        dataset_kwargs_list=[
            dict(name=SO101_DATASET_PICKUP,     data_dir=FLAGS.data_dir, **_dataset_kwargs),
            dict(name=SO101_DATASET_PUT_INSIDE, data_dir=FLAGS.data_dir, **_dataset_kwargs),
        ],
        sample_weights=[0.5, 0.5],             
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

    # 텍스트 토크나이저 처리
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

    # proprio tokenizer 추가 (관절 상태 입력, degree 단위 ±180)
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-180.0,
        high=180.0,
        obs_keys=["proprio"],
    )

    # [수정 3] Action Head 교체 로직(L1ActionHead) 삭제
    # 이 코드를 지움으로써 사전학습된 강력한 Diffusion Head와 그 가중치를 그대로 보존합니다.

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
    
    # Diffusion Head를 지우지 않았으므로, pretrained_model의 가중치가 
    # Action Head 영역까지 성공적으로 매핑되어 병합됩니다.
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    # ---------------------------------------------------------------
    # 5. 옵티마이저 & 학습 상태 설정
    # ---------------------------------------------------------------
    # [수정 4] 거대한 트랜스포머의 기존 지식 붕괴를 막기 위해 LR을 1e-5로 낮추고 웜업 축소
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 1e-5, 200), optax.constant_schedule(1e-5)], [200]
    )
    tx = optax.adamw(learning_rate)
    
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
        
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

    # [수정 5] 100 에피소드 극소량 데이터의 공간적 과적합 방지를 위해 스텝 수 대폭 감소
    # Batch 8 기준 약 3000 스텝이면 충분히 환경에 적응합니다.
    NUM_TRAIN_STEPS = 3000
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
    
    # 최종 체크포인트 저장
    train_state.model.save_pretrained(
        step=NUM_TRAIN_STEPS - 1, checkpoint_path=FLAGS.save_dir
    )
    logging.info("Finetuning complete! Ready for evaluation.")


if __name__ == "__main__":
    app.run(main)