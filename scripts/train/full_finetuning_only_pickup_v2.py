"""
SO-101 Octo 파인튜닝 스크립트 v2 (Pickup single-task, trim + frozen vision + val tracking)
===========================================================================================
기존 full_finetuning_only_pickup.py 기반.

v2의 핵심 변경:
  1. Dataset: so101_pickup → so101_pickup_v2 (trim 적용 + train/val split)
  2. Frozen: T5만 → T5 + Vision encoder (SmallStem16) 추가 freeze
     → 학습 파라미터 ~80% → ~30%로 축소, vision 일반성 보존
  3. Val loss tracking: train/val 둘 다 매 N step마다 측정 후 wandb log
  4. Best val ckpt 저장: val loss가 가장 낮은 시점의 ckpt 별도 보존
  5. NUM_TRAIN_STEPS: 2000 → 5000 (val curve로 best step 선정용 여유)

학습 설정 (기존 유지):
  - Batch size: 8 (24GB VRAM 권장)
  - LR: 1e-5, warmup 100 step + constant
  - Action head: pretrained DiffusionActionHead 그대로 (교체 안 함)
"""
from absl import app, flags, logging
import os
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

flags.DEFINE_string("pretrained_path", None, "Path to pre-trained Octo checkpoint directory.")
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 8, "Batch size for finetuning (8 recommended for 24GB VRAM).")
flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen (extra to vision).",
)

# ===================================================================
# SO-101 전용 설정 (v2)
# ===================================================================
SO101_ACTION_DIM         = 7
SO101_ACTION_HORIZON     = 4
SO101_DATASET_PICKUP_V2  = "so101_pickup_v2"   # [v2] trim + train/val split 적용

# 학습 설정
NUM_TRAIN_STEPS = 5000     # val curve 충분히 보기 위해 5000으로 늘림
LOG_INTERVAL    = 50       # train loss + 기타 metric 로깅 주기
VAL_INTERVAL    = 100      # val loss 측정 주기 (jit 컴파일 부담 줄이려 train보다 드물게)
SAVE_INTERVAL   = 500      # 정기 ckpt 저장 주기
VAL_BATCHES     = 8        # val loss 평균낼 batch 수 (variance 줄이기)

# Frozen 전략
# - 기본: T5 (hf_model) — pretrained config의 frozen_keys 그대로
# - 추가: SmallStem16 vision encoder들 (image_primary용 + image_wrist용)
# - 옵션: --freeze_transformer=True 시 BlockTransformer도 freeze (action head만 학습)
EXTRA_FROZEN_PATTERNS = ["*SmallStem*"]
# ===================================================================


def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")

    wandb.init(name="finetune_so101_pickup_v2_frozen_260515", project="octo")

    # ---------------------------------------------------------------
    # 1. 사전학습 모델 로드
    # ---------------------------------------------------------------
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # ---------------------------------------------------------------
    # 2. 파인튜닝 데이터셋 로드 (Train + Val 둘 다)
    # ---------------------------------------------------------------
    logging.info("Loading finetuning dataset (train + val splits)...")

    _dataset_kwargs = dict(
        image_obs_keys={
            "primary": "image_primary",
            "wrist":   "image_wrist",
        },
        proprio_obs_key="state",
        language_key="language_instruction",
    )

    # Train dataset
    train_dataset = make_interleaved_dataset(
        dataset_kwargs_list=[
            dict(name=SO101_DATASET_PICKUP_V2, data_dir=FLAGS.data_dir, **_dataset_kwargs),
        ],
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
        train_dataset.repeat().batch(FLAGS.batch_size).iterator()
    )

    # Val dataset
    # train=False: shuffle 없음, augmentation 없음 (deterministic val loss 위해)
    val_dataset = make_interleaved_dataset(
        dataset_kwargs_list=[
            dict(name=SO101_DATASET_PICKUP_V2, data_dir=FLAGS.data_dir, **_dataset_kwargs),
        ],
        shuffle_buffer_size=1000,
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
        train=False,
    )
    val_data_iter = (
        val_dataset.repeat().batch(FLAGS.batch_size).iterator()
    )

    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    val_data_iter   = map(process_batch, val_data_iter)
    example_batch   = next(train_data_iter)

    # ---------------------------------------------------------------
    # 3. 모델 설정 수정 (proprio tokenizer 추가만, action head는 그대로)
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
        dataset_statistics=train_dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    # ---------------------------------------------------------------
    # 5. 옵티마이저 & 학습 상태 설정 (v2: vision encoder 추가 freeze)
    # ---------------------------------------------------------------
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 1e-5, 100), optax.constant_schedule(1e-5)], [100]
    )
    tx = optax.adamw(learning_rate)

    # Frozen keys: 기본(T5) + Vision encoder + (옵션) Transformer
    frozen_keys = list(model.config.get("optimizer", {}).get("frozen_keys", []))
    frozen_keys.extend(EXTRA_FROZEN_PATTERNS)  # *SmallStem* 추가
    if FLAGS.freeze_transformer:
        frozen_keys.append("octo_transformer.*")
        logging.info("❄️ Transformer backbone also frozen. Training Action Head ONLY.")
    logging.info(f"❄️ Final frozen_keys: {frozen_keys}")

    tx = freeze_weights(tx, model.params, frozen_keys)

    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # ---------------------------------------------------------------
    # 6. Loss + Train step + Val step
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

    @jax.jit
    def val_step(state, batch):
        # train=False: dropout 비활성, augmentation은 frame_transform에서 train=False로 이미 비활성
        rng, dropout_rng = jax.random.split(state.rng)
        loss, info = loss_fn(state.model.params, batch, dropout_rng, train=False)
        return loss, info

    def eval_val_loss(state, n_batches=VAL_BATCHES):
        """val_data_iter에서 n_batches만큼 평균 loss 측정."""
        losses = []
        for _ in range(n_batches):
            batch = next(val_data_iter)
            loss, _ = val_step(state, batch)
            losses.append(float(loss))
        return sum(losses) / len(losses)

    # ---------------------------------------------------------------
    # 7. 학습 루프 (val loss tracking + best ckpt 저장)
    # ---------------------------------------------------------------
    best_val_loss = float("inf")
    best_val_step = -1
    best_ckpt_dir = os.path.join(FLAGS.save_dir, "best_val")

    logging.info(f"Starting finetuning ({NUM_TRAIN_STEPS} steps)...")
    for i in tqdm.tqdm(range(NUM_TRAIN_STEPS), total=NUM_TRAIN_STEPS, dynamic_ncols=True):
        batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)

        # Train metric logging
        if (i + 1) % LOG_INTERVAL == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )

        # Val loss 측정 + best ckpt 저장
        if (i + 1) % VAL_INTERVAL == 0:
            val_loss = eval_val_loss(train_state)
            wandb.log({"validation/loss": val_loss}, step=i)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_step = i
                # Best val ckpt 별도 저장
                train_state.model.save_pretrained(
                    step=i, checkpoint_path=best_ckpt_dir
                )
                logging.info(f"[Step {i+1}] 🏆 New best val_loss={val_loss:.5f} → saved to {best_ckpt_dir}")
            else:
                logging.info(f"[Step {i+1}] val_loss={val_loss:.5f} (best={best_val_loss:.5f} @ step {best_val_step+1})")

        # 정기 ckpt 저장
        if (i + 1) % SAVE_INTERVAL == 0:
            train_state.model.save_pretrained(
                step=i, checkpoint_path=FLAGS.save_dir
            )

    # 최종 ckpt 저장
    train_state.model.save_pretrained(
        step=NUM_TRAIN_STEPS - 1, checkpoint_path=FLAGS.save_dir
    )
    logging.info(f"Finetuning complete!")
    logging.info(f"🏆 Best val_loss = {best_val_loss:.5f} at step {best_val_step+1}")
    logging.info(f"📁 Best ckpt: {best_ckpt_dir}")
    logging.info(f"📁 Final ckpt: {FLAGS.save_dir}/step_{NUM_TRAIN_STEPS-1}")

    wandb.log({"final/best_val_loss": best_val_loss, "final/best_val_step": best_val_step}, step=NUM_TRAIN_STEPS-1)


if __name__ == "__main__":
    app.run(main)
