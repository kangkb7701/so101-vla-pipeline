"""
SO-101 Octo fine-tuning on VP-SAM3 pick-and-place data with binary gripper actions.

Training policy:
  - Dataset: so101_pick_place_vp_sam3_binary_gripper TFDS/RLDS
  - Inputs: overlay top image + wrist image + proprio + language
  - Actions: 7D [dx, dy, dz, drx, dry, drz, gripper_open_binary]
  - Action horizon/chunk: 4 (matches Octo pretrained head)
  - Frozen modules: T5 language encoder + wrist image tokenizer by default
  - Trainable modules: top image tokenizer + Octo transformer + proprio tokenizer + action head
  - Optional weighted diffusion loss for binary gripper timing
"""

from absl import app, flags, logging
import copy
import os

import flax
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tqdm
import wandb
from einops import rearrange

from octo.data.dataset import make_interleaved_dataset
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    TrainState,
    freeze_weights,
    merge_params,
    process_text,
)


FLAGS = flags.FLAGS

DEFAULT_PRETRAINED_PATH = "hf://rail-berkeley/octo-base-1.5"

flags.DEFINE_string(
    "pretrained_path",
    DEFAULT_PRETRAINED_PATH,
    "Path or HF URI for the pretrained Octo checkpoint.",
)
flags.DEFINE_string("data_dir", "/home/aivlab/tensorflow_datasets", "TFDS data_dir.")
flags.DEFINE_string("dataset", "so101_pick_place_vp_sam3_binary_gripper", "TFDS dataset name.")
flags.DEFINE_string("save_dir", None, "Directory for saving fine-tuned checkpoints.")
flags.DEFINE_integer("batch_size", 8, "Batch size for horizon=4 on RTX 3090.")
flags.DEFINE_integer("num_train_steps", 2000, "Number of fine-tuning steps.")
flags.DEFINE_integer("log_interval", 50, "Train metric logging interval.")
flags.DEFINE_integer("val_interval", 100, "Validation interval.")
flags.DEFINE_integer("save_interval", 500, "Regular checkpoint interval.")
flags.DEFINE_integer("val_batches", 12, "Number of validation batches to average.")
flags.DEFINE_float("learning_rate", 3e-6, "Constant LR after warmup.")
flags.DEFINE_integer("warmup_steps", 200, "Linear warmup steps.")
flags.DEFINE_integer("early_stop_patience", 0, "0 disables early stopping.")
flags.DEFINE_bool("wandb", True, "Enable Weights & Biases logging.")
flags.DEFINE_string("wandb_project", "octo", "wandb project name.")
flags.DEFINE_string("wandb_run_name", "vp_sam3_pick_place_h4_binary_gripper", "wandb run name.")
flags.DEFINE_bool("freeze_primary_vision", False, "Freeze top/primary image tokenizer and ViT.")
flags.DEFINE_bool("freeze_wrist_vision", True, "Freeze wrist image tokenizer and ViT.")
flags.DEFINE_float("gripper_loss_weight", 2.0, "Extra diffusion loss weight for gripper action dimension.")
flags.DEFINE_float("close_gripper_loss_weight", 2.0, "Additional gripper-dim weight when target gripper action is close (0).")


SO101_DATASET = "so101_pick_place_vp_sam3_binary_gripper"
SO101_ACTION_DIM = 7
SO101_ACTION_HORIZON = 4
WINDOW_SIZE = 2

IMAGE_RESIZE = {
    "primary": (256, 256),
    "wrist": (128, 128),
}

EXTRA_FROZEN_PATTERNS = []


def _require_flags():
    if FLAGS.save_dir is None:
        raise ValueError("--save_dir is required")



def _weighted_diffusion_action_loss(action_head, transformer_embeddings, actions, timestep_pad_mask, action_pad_mask, train=True):
    batch_size, window_size = timestep_pad_mask.shape
    actions_flat = rearrange(actions, "b w h a -> b w (h a)")
    actions_flat = jnp.clip(actions_flat, -action_head.max_action, action_head.max_action)

    rng = action_head.make_rng("dropout")
    time_key, noise_key = jax.random.split(rng)
    time = jax.random.randint(
        time_key,
        (action_head.n_diffusion_samples, batch_size, window_size, 1),
        0,
        action_head.diffusion_steps,
    )
    noise = jax.random.normal(
        noise_key, (action_head.n_diffusion_samples,) + actions_flat.shape
    )

    scale = jnp.sqrt(action_head.alpha_hats[time])
    std = jnp.sqrt(1 - action_head.alpha_hats[time])
    noisy_actions = scale * actions_flat[None] + std * noise
    pred_eps = action_head(
        transformer_embeddings, train=train, time=time, noisy_actions=noisy_actions
    )

    valid = timestep_pad_mask[:, :, None, None] & action_pad_mask
    weights = jnp.ones_like(actions, dtype=jnp.float32)

    dim_weights = jnp.ones((SO101_ACTION_DIM,), dtype=jnp.float32)
    dim_weights = dim_weights.at[6].set(FLAGS.gripper_loss_weight)
    weights = weights * dim_weights

    close_target = actions[..., 6] <= 0.5
    close_multiplier = jnp.where(close_target, FLAGS.close_gripper_loss_weight, 1.0)
    weights = weights.at[..., 6].multiply(close_multiplier)

    weights = rearrange(weights * valid.astype(jnp.float32), "b w h a -> b w (h a)")[None]
    err = jnp.square(pred_eps - noise)
    loss = jnp.sum(err * weights) / jnp.clip(jnp.sum(weights), a_min=1e-5)
    mse = jnp.sum(err * (weights > 0)) / jnp.clip(jnp.sum(weights > 0), a_min=1e-5)

    gripper_err = err[..., 6::SO101_ACTION_DIM]
    gripper_weights = weights[..., 6::SO101_ACTION_DIM]
    gripper_mse = jnp.sum(gripper_err * (gripper_weights > 0)) / jnp.clip(
        jnp.sum(gripper_weights > 0), a_min=1e-5
    )

    loss = loss * SO101_ACTION_DIM
    return loss, {
        "loss": loss,
        "mse": mse * SO101_ACTION_DIM,
        "gripper_mse": gripper_mse,
        "gripper_loss_weight": jnp.asarray(FLAGS.gripper_loss_weight, dtype=jnp.float32),
        "close_gripper_loss_weight": jnp.asarray(FLAGS.close_gripper_loss_weight, dtype=jnp.float32),
    }

def _set_action_head_horizon(config):
    action_head = config["model"]["heads"]["action"]
    if not isinstance(action_head, dict) or "kwargs" not in action_head:
        raise TypeError(f"Unexpected action head spec: {action_head}")

    action_head["kwargs"]["action_dim"] = SO101_ACTION_DIM
    action_head["kwargs"]["action_horizon"] = SO101_ACTION_HORIZON

    logging.info(
        "Action head configured: %s action_dim=%d action_horizon=%d",
        action_head.get("name"),
        SO101_ACTION_DIM,
        SO101_ACTION_HORIZON,
    )


def main(_):
    _require_flags()
    assert FLAGS.batch_size % jax.device_count() == 0, (
        "Batch size must be divisible by jax.device_count()."
    )

    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")

    if FLAGS.wandb:
        wandb.init(name=FLAGS.wandb_run_name, project=FLAGS.wandb_project)
    else:
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")

    logging.info("Loading pretrained Octo model: %s", FLAGS.pretrained_path)
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    dataset_kwargs = dict(
        image_obs_keys={
            "primary": "image_primary",
            "wrist": "image_wrist",
        },
        proprio_obs_key="state",
        language_key="language_instruction",
        # Match Octo/OXE convention: normalize EE deltas, keep binary gripper in [0, 1].
        action_normalization_mask=[True, True, True, True, True, True, False],
    )
    traj_transform_kwargs = dict(
        window_size=WINDOW_SIZE,
        action_horizon=SO101_ACTION_HORIZON,
    )
    frame_transform_kwargs = dict(resize_size=IMAGE_RESIZE)

    logging.info("Loading train split from TFDS dataset: %s", FLAGS.dataset)
    train_dataset = make_interleaved_dataset(
        dataset_kwargs_list=[
            dict(name=FLAGS.dataset, data_dir=FLAGS.data_dir, **dataset_kwargs),
        ],
        shuffle_buffer_size=10000,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        train=True,
    )
    train_data_iter = train_dataset.repeat().batch(FLAGS.batch_size).iterator()

    logging.info("Loading val split from TFDS dataset: %s", FLAGS.dataset)
    val_dataset = make_interleaved_dataset(
        dataset_kwargs_list=[
            dict(name=FLAGS.dataset, data_dir=FLAGS.data_dir, **dataset_kwargs),
        ],
        shuffle_buffer_size=1000,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        train=False,
    )
    val_data_iter = val_dataset.repeat().batch(FLAGS.batch_size).iterator()

    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    val_data_iter = map(process_batch, val_data_iter)
    example_batch = next(train_data_iter)

    logging.info("Example action shape: %s", example_batch["action"].shape)
    if example_batch["action"].shape[-2:] != (SO101_ACTION_HORIZON, SO101_ACTION_DIM):
        raise ValueError(
            f"Expected action shape [..., {SO101_ACTION_HORIZON}, {SO101_ACTION_DIM}], "
            f"got {example_batch['action'].shape}"
        )

    config = copy.deepcopy(pretrained_model.config)

    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-180.0,
        high=180.0,
        obs_keys=["proprio"],
    )
    _set_action_head_horizon(config)

    logging.info("Initializing Octo model from modified config...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=train_dataset.dataset_statistics,
    )

    logging.info("Merging compatible pretrained parameters...")
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    learning_rate = optax.join_schedules(
        [
            optax.linear_schedule(0.0, FLAGS.learning_rate, FLAGS.warmup_steps),
            optax.constant_schedule(FLAGS.learning_rate),
        ],
        [FLAGS.warmup_steps],
    )
    tx = optax.adamw(learning_rate)

    frozen_keys = list(model.config.get("optimizer", {}).get("frozen_keys", []))
    frozen_keys.extend(EXTRA_FROZEN_PATTERNS)
    if FLAGS.freeze_primary_vision:
        frozen_keys.append("*observation_tokenizers_primary*")
    if FLAGS.freeze_wrist_vision:
        frozen_keys.append("*observation_tokenizers_wrist*")

    logging.info("Final frozen_keys: %s", frozen_keys)
    logging.info(
        "Trainable: transformer + proprio tokenizer + action head + %s primary vision; frozen wrist vision=%s",
        "frozen" if FLAGS.freeze_primary_vision else "trainable",
        FLAGS.freeze_wrist_vision,
    )
    tx = freeze_weights(tx, model.params, frozen_keys)

    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = _weighted_diffusion_action_loss(
            bound_module.heads["action"],
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

    def eval_val_loss(state, n_batches=FLAGS.val_batches):
        losses = []
        for _ in range(n_batches):
            batch = next(val_data_iter)
            loss, _ = val_step(state, batch)
            losses.append(float(loss))
        return sum(losses) / len(losses)

    best_val_loss = float("inf")
    best_val_step = -1
    best_ckpt_dir = os.path.join(FLAGS.save_dir, "best_val")
    patience_count = 0
    early_stopped = False

    logging.info("Starting fine-tuning for %d steps...", FLAGS.num_train_steps)
    for i in tqdm.tqdm(range(FLAGS.num_train_steps), total=FLAGS.num_train_steps, dynamic_ncols=True):
        batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)

        if (i + 1) % FLAGS.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )

        if (i + 1) % FLAGS.val_interval == 0:
            val_loss = eval_val_loss(train_state)
            wandb.log({"validation/loss": val_loss}, step=i)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_step = i
                train_state.model.save_pretrained(step=i, checkpoint_path=best_ckpt_dir)
                logging.info(
                    "[Step %d] New best val_loss=%.6f -> %s",
                    i + 1,
                    val_loss,
                    best_ckpt_dir,
                )
                patience_count = 0
            else:
                patience_count += FLAGS.val_interval
                logging.info(
                    "[Step %d] val_loss=%.6f best=%.6f@%d patience=%s/%s",
                    i + 1,
                    val_loss,
                    best_val_loss,
                    best_val_step + 1,
                    patience_count,
                    FLAGS.early_stop_patience if FLAGS.early_stop_patience > 0 else "off",
                )

            if FLAGS.early_stop_patience > 0 and patience_count >= FLAGS.early_stop_patience:
                early_stopped = True
                logging.info("Early stopping at step %d", i + 1)
                break

        if (i + 1) % FLAGS.save_interval == 0:
            train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)

    final_step = i if early_stopped else FLAGS.num_train_steps - 1
    train_state.model.save_pretrained(step=final_step, checkpoint_path=FLAGS.save_dir)
    logging.info("Fine-tuning complete.")
    logging.info("Best val_loss=%.6f at step %d", best_val_loss, best_val_step + 1)
    logging.info("Best checkpoint: %s", best_ckpt_dir)
    logging.info("Final checkpoint dir: %s", FLAGS.save_dir)

    wandb.log(
        {
            "final/best_val_loss": best_val_loss,
            "final/best_val_step": best_val_step,
            "final/total_steps": final_step + 1,
            "final/early_stopped": int(early_stopped),
        },
        step=final_step,
    )


if __name__ == "__main__":
    app.run(main)
