"""
SO-101 Octo 파인튜닝 스크립트 — checkpoint4 방식 + 50ep + trim + val tracking
=========================================================================
checkpoint4의 freezing 전략(T5만 frozen, vision+transformer 모두 학습)을
그대로 적용하면서, v2의 인프라(trim + train/val split + best ckpt 자동 선정)를
추가한 ablation 실험용 스크립트.

[checkpoint4와 비교]
  공통:
    - frozen_keys: ['*hf_model*'] (T5만 frozen)
    - vision encoder, transformer, action head 모두 학습
    - DiffusionActionHead 그대로 (모듈 교체 없음)
    - batch_size=8, LR=1e-5, warmup 100
    - 2000 step
  차이:
    - 데이터: pickup → pickup_v2_50ep (trim 적용)
    - val tracking + best ckpt 자동 저장 추가
    - Early stopping (patience 기반) 옵션 추가

[v2와 비교]
  공통: trim, val tracking, best ckpt 저장 인프라
  차이:
    - frozen_keys: T5+SmallStem(v2) → T5만(이번)  ← vision 학습됨
    - 데이터: 100ep → 50ep
    - step: 5000 → 2000 (checkpoint4와 동일)
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
flags.DEFINE_integer("batch_size", 8, "Batch size for finetuning (8 for 24GB VRAM).")

# ===================================================================
# SO-101 전용 설정
# ===================================================================
SO101_ACTION_DIM      = 7
SO101_ACTION_HORIZON  = 4
SO101_DATASET         = "so101_pickup_v2_50ep"   # [new] 50ep 데이터셋 (40 train + 10 val)

# 학습 설정 (checkpoint4와 동일)
NUM_TRAIN_STEPS = 2000
LOG_INTERVAL    = 50      # train metric 로깅
VAL_INTERVAL    = 100     # val loss 측정
SAVE_INTERVAL   = 500     # 정기 ckpt 저장
VAL_BATCHES     = 16      # val loss 평균낼 batch 수 (v2의 8 → 16으로 noise ↓)

# Early stopping (val_loss가 patience step 동안 개선 없으면 중단)
#   0 = 비활성 (학습 끝까지 진행하고 best_val ckpt만 자동 저장)
#   500 = 500 step 동안 개선 없으면 중단
EARLY_STOP_PATIENCE = 0

# Frozen 전략: checkpoint4 방식 — T5만 frozen, 나머지 모두 학습
# pretrained config의 frozen_keys=['*hf_model*']를 그대로 사용 → 추가 패턴 없음
EXTRA_FROZEN_PATTERNS = []
# ===================================================================


def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")

    wandb.init(name="finetune_so101_pickup_v2_50ep_unfrozen_260518", project="octo")

    # ---------------------------------------------------------------
    # 1. Pre-trained model 로드
    # ---------------------------------------------------------------
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # ---------------------------------------------------------------
    # 2. Train + Val 데이터셋 로드
    # ---------------------------------------------------------------
    logging.info(f"Loading finetuning dataset ({SO101_DATASET}, train + val splits)...")

    _dataset_kwargs = dict(
        image_obs_keys={
            "primary": "image_primary",
            "wrist":   "image_wrist",
        },
        proprio_obs_key="state",
        language_key="language_instruction",
    )

    train_dataset = make_interleaved_dataset(
        dataset_kwargs_list=[
            dict(name=SO101_DATASET, data_dir=FLAGS.data_dir, **_dataset_kwargs),
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

    val_dataset = make_interleaved_dataset(
        dataset_kwargs_list=[
            dict(name=SO101_DATASET, data_dir=FLAGS.data_dir, **_dataset_kwargs),
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
    # 3. 모델 설정 수정 (proprio tokenizer만 추가, action head는 그대로)
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
    # 4. 모델 초기화 + pretrained weight merge
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
    # 5. 옵티마이저 + 학습 상태 (checkpoint4와 동일 LR/warmup)
    # ---------------------------------------------------------------
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 1e-5, 100), optax.constant_schedule(1e-5)], [100]
    )
    tx = optax.adamw(learning_rate)

    # Frozen keys: T5만 (checkpoint4 방식)
    frozen_keys = list(model.config.get("optimizer", {}).get("frozen_keys", []))
    if EXTRA_FROZEN_PATTERNS:
        frozen_keys.extend(EXTRA_FROZEN_PATTERNS)
    logging.info(f"❄️ Final frozen_keys: {frozen_keys}")
    logging.info(f"🔥 학습되는 모듈: vision encoder + transformer + proprio + action head")

    tx = freeze_weights(tx, model.params, frozen_keys)

    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # ---------------------------------------------------------------
    # 6. Loss + train_step + val_step
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
    # 7. 학습 루프 (val tracking + best ckpt + early stopping)
    # ---------------------------------------------------------------
    best_val_loss   = float("inf")
    best_val_step   = -1
    best_ckpt_dir   = os.path.join(FLAGS.save_dir, "best_val")
    patience_count  = 0
    early_stopped   = False

    logging.info(f"Starting finetuning ({NUM_TRAIN_STEPS} steps)...")
    for i in tqdm.tqdm(range(NUM_TRAIN_STEPS), total=NUM_TRAIN_STEPS, dynamic_ncols=True):
        batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)

        # Train metric 로깅
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
                train_state.model.save_pretrained(
                    step=i, checkpoint_path=best_ckpt_dir
                )
                logging.info(f"[Step {i+1}] 🏆 New best val_loss={val_loss:.5f} → saved to {best_ckpt_dir}")
                patience_count = 0
            else:
                patience_count += VAL_INTERVAL
                logging.info(
                    f"[Step {i+1}] val_loss={val_loss:.5f} "
                    f"(best={best_val_loss:.5f} @ step {best_val_step+1}, "
                    f"patience={patience_count}/{EARLY_STOP_PATIENCE if EARLY_STOP_PATIENCE>0 else 'off'})"
                )

            # Early stopping
            if EARLY_STOP_PATIENCE > 0 and patience_count >= EARLY_STOP_PATIENCE:
                logging.info(
                    f"⏹ Early stop at step {i+1} "
                    f"(val_loss 개선 없이 {patience_count} step 경과)"
                )
                early_stopped = True
                break

        # 정기 ckpt
        if (i + 1) % SAVE_INTERVAL == 0:
            train_state.model.save_pretrained(
                step=i, checkpoint_path=FLAGS.save_dir
            )

    # 최종 ckpt
    final_step = i if early_stopped else NUM_TRAIN_STEPS - 1
    train_state.model.save_pretrained(
        step=final_step, checkpoint_path=FLAGS.save_dir
    )

    logging.info(f"Finetuning complete!{' (early stopped)' if early_stopped else ''}")
    logging.info(f"🏆 Best val_loss = {best_val_loss:.5f} at step {best_val_step+1}")
    logging.info(f"📁 Best ckpt: {best_ckpt_dir}")
    logging.info(f"📁 Final ckpt: {FLAGS.save_dir}/step_{final_step}")

    wandb.log({
        "final/best_val_loss": best_val_loss,
        "final/best_val_step": best_val_step,
        "final/total_steps":   final_step + 1,
        "final/early_stopped": int(early_stopped),
    }, step=final_step)


if __name__ == "__main__":
    app.run(main)
