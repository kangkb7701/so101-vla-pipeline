"""
SO-101 Octo 파인튜닝 스크립트
==============================
so101_dataset_builder.py로 변환한 RLDS 데이터셋을 사용하여
Octo를 SO-101 로봇에 맞게 파인튜닝합니다.

원본 02_finetune_new_observation_action.py (ALOHA 기준)에서 변경된 사항:
  - action_dim: 14 → 6 (SO-101 6축)
  - action_horizon: 50 → 4 (Octo 기본값, pick 태스크에 적합)
  - image_obs_keys: top 카메라 → primary, front 카메라 → wrist
  - wrist tokenizer 삭제하지 않음 (front 카메라를 wrist로 사용)
  - proprio tokenizer 추가

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
from octo.model.components.action_heads import L1ActionHead
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
flags.DEFINE_integer("batch_size", 128, "Batch size for finetuning.")
flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)


# ===================================================================
# SO-101 전용 설정
# ===================================================================
SO101_ACTION_DIM = 7        # FK delta: [dx, dy, dz, drx, dry, drz, d_gripper]
                            # (pretrained Bridge 데이터셋과 동일한 7-dim EE delta 포맷)
SO101_ACTION_HORIZON = 4    # Octo 기본 action chunk 길이
                            # (ALOHA는 50이지만, 단순 pick 태스크에는 4가 적절)
SO101_DATASET_PICKUP     = "so101_pickup"       # pickup 태스크 tfds 이름
SO101_DATASET_PUT_INSIDE = "so101_put_inside"   # put_inside 태스크 tfds 이름
# ===================================================================


def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # TensorFlow가 GPU 메모리를 점유하지 않도록 설정 (데이터 로딩에만 사용)
    tf.config.set_visible_devices([], "GPU")

    # wandb 로깅 설정
    wandb.init(name="finetune_so101", project="octo")

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
        sample_weights=[0.5, 0.5],             # ★ 두 태스크 균등 샘플링
        shuffle_buffer_size=10000,             # ★ Octo 최신 버전 요구사항 추가
        traj_transform_kwargs=dict(
            window_size=2,                     # Octo 기본: 2 타임스텝 관찰 이력
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
    logging.info("Updating model for SO-101 observation & action space...")
    config = pretrained_model.config

    # ★ wrist tokenizer를 삭제하지 않음! (front 카메라를 wrist로 사용)
    # 원본 ALOHA 예시에서는 del config["model"]["observation_tokenizers"]["wrist"]
    # 이 부분을 하지 않습니다.

    # ★ proprio tokenizer 추가 (관절 상태 입력)
    # ★ low/high 범위: SO-101 관절값은 degree 단위 (-161° ~ +99°)
    #   action 포함 전체 극단값 ±161°에 여유를 두어 ±180으로 설정
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-180.0,
        high=180.0,
        obs_keys=["proprio"],
    )

    # ★ 액션 헤드를 SO-101 사양으로 교체
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        action_horizon=SO101_ACTION_HORIZON,   # 4스텝 action chunk
        action_dim=SO101_ACTION_DIM,           # 6축 관절
        readout_key="readout_action",
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
    # 기존 사전학습 가중치 중 호환되는 것을 병합
    # proprio tokenizer와 새 action head의 가중치는 랜덤 초기화 상태 유지
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    # ---------------------------------------------------------------
    # 5. 옵티마이저 & 학습 상태 설정
    # ---------------------------------------------------------------
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 600), optax.constant_schedule(3e-5)], [600]
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

    # ★ 100 에피소드 × ~110 프레임(10fps) ≈ 11000 프레임
    #   Batch 16 기준 약 40 Epoch = 28000 스텝
    NUM_TRAIN_STEPS = 28000
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 1000

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
    logging.info("Finetuning complete!")


if __name__ == "__main__":
    app.run(main)
