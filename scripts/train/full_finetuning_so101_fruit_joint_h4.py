"""Fine-tune Octo-Base 1.5 on SO-101 fruit data with 6D joint-space actions.

Identical recipe to full_finetuning_so101_fruit_raw_h4_binary.py (T5 + SmallStem
frozen; transformer, proprio tokenizer, action head trainable). Only the action
head changes: 5 arm joints (relative or absolute, see the dataset builder) plus the
gripper command in degrees (or a binary flag with --binary_gripper), so it is
re-initialised from scratch with action_dim=6.
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
from octo.utils.train_utils import TrainState, freeze_weights, merge_params, process_text


FLAGS = flags.FLAGS

flags.DEFINE_string("pretrained_path", "hf://rail-berkeley/octo-base-1.5", "Octo checkpoint.")
flags.DEFINE_string("data_dir", "/home/aivlab/tensorflow_datasets", "TFDS data directory.")
flags.DEFINE_string("dataset", "so101_fruit_joint_rel_cmd_h4", "TFDS dataset name.")
flags.DEFINE_string("save_dir", None, "Checkpoint output directory.")
flags.DEFINE_integer("batch_size", 4, "Batch size for Octo-Base on a 24GB GPU.")
flags.DEFINE_integer("num_train_steps", 8000, "Maximum fine-tuning steps.")
flags.DEFINE_integer("log_interval", 50, "Training metric interval.")
flags.DEFINE_integer("val_interval", 250, "Validation interval.")
flags.DEFINE_integer("save_interval", 1000, "Regular checkpoint interval.")
flags.DEFINE_integer("val_batches", 32, "Validation batches per evaluation.")
# The 6D action head has no pretrained counterpart (octo-base-1.5 ships a 7D head,
# so merge_params keeps our random init). 1e-6 — fine for nudging a pretrained head —
# left it at its init norm after 8000 steps and the model predicted noise. These are
# Octo's own finetune_config defaults for a freshly initialised head.
flags.DEFINE_float("learning_rate", 3e-4, "Peak learning rate for the new head + proprio path.")
flags.DEFINE_float("end_learning_rate", 3e-6, "Final cosine-decay learning rate.")
flags.DEFINE_float(
    "backbone_learning_rate",
    1e-5,
    "Peak learning rate for the pretrained Octo transformer. Training it at the head's "
    "3e-4 widened the predictions; freezing it entirely cost the lift/elbow correlation. "
    "Set equal to --learning_rate for a single-rate run, or 0 to freeze.",
)
flags.DEFINE_integer(
    "grad_accum_steps",
    8,
    "Micro-batches accumulated per update. batch_size=4 fills only ~half the GPU and "
    "makes the diffusion gradient noisy, which shows up as over-wide predictions; this "
    "raises the effective batch without extra VRAM.",
)
flags.DEFINE_integer("warmup_steps", 500, "Linear warmup steps.")
flags.DEFINE_float("weight_decay", 0.01, "AdamW weight decay.")
flags.DEFINE_float("clip_gradient", 1.0, "Global gradient norm clipping.")
flags.DEFINE_integer("early_stop_patience", 0, "0 disables early stopping.")
flags.DEFINE_bool(
    "binary_gripper",
    False,
    "Dataset gripper is a binary open flag: skip its normalization and apply the "
    "binary loss weights below. Default: continuous gripper command in degrees.",
)
flags.DEFINE_float("gripper_loss_weight", 2.0, "Binary gripper dimension loss weight.")
flags.DEFINE_float("close_gripper_loss_weight", 1.5, "Extra weight for close=0 targets.")
flags.DEFINE_bool(
    "freeze_backbone",
    False,
    "Freeze the whole pretrained Octo transformer (incl. language/vision tokenizers); "
    "train only the new proprio tokenizer and the 6D action head.",
)
flags.DEFINE_integer("shuffle_buffer_size", 3000, "Train shuffle buffer in frames.")
flags.DEFINE_integer(
    "data_threads",
    4,
    "Trajectory read/transform parallelism. Octo's default (-1, autotune) keeps ~17 GB "
    "of decoded trajectories in flight per pipeline; 4 threads keep ~11 GB.",
)
flags.DEFINE_integer(
    "reset_iter_every",
    1000,
    "Rebuild the train iterator every N steps. tf.data's shuffle stage in this pipeline "
    "leaks ~0.75 MB per consumed frame (measured; buffer size irrelevant) and two runs "
    "were OOM-killed at ~43 GB. Recreating the iterator releases it.",
)
flags.DEFINE_bool("wandb", True, "Enable W&B.")
flags.DEFINE_string("wandb_project", "octo", "W&B project.")
flags.DEFINE_string("wandb_run_name", "so101_fruit_joint_rel_cmd_base_h4", "W&B run.")

ACTION_DIM = 6
GRIPPER_INDEX = ACTION_DIM - 1
ACTION_HORIZON = 4
WINDOW_SIZE = 2
IMAGE_RESIZE = {"primary": (256, 256), "wrist": (128, 128)}

IMAGE_AUGMENT = {
    "primary": {
        "random_resized_crop": {"scale": [0.9, 1.0], "ratio": [0.95, 1.05]},
        "random_brightness": [0.1],
        "random_contrast": [0.9, 1.1],
        "random_saturation": [0.9, 1.1],
        "random_hue": [0.05],
        "augment_order": [
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    },
    "wrist": {
        "random_resized_crop": {"scale": [0.9, 1.0], "ratio": [0.95, 1.05]},
        "random_brightness": [0.1],
        "random_contrast": [0.9, 1.1],
        "random_saturation": [0.9, 1.1],
        "random_hue": [0.05],
        "augment_order": [
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    },
}


def weighted_action_loss(
    action_head,
    transformer_embeddings,
    actions,
    timestep_pad_mask,
    action_pad_mask,
    train=True,
):
    batch_size, window_size = timestep_pad_mask.shape
    actions_flat = rearrange(actions, "b w h a -> b w (h a)")
    actions_flat = jnp.clip(actions_flat, -action_head.max_action, action_head.max_action)
    time_key, noise_key = jax.random.split(action_head.make_rng("dropout"))
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
    pred_eps = action_head(
        transformer_embeddings,
        train=train,
        time=time,
        noisy_actions=scale * actions_flat[None] + std * noise,
    )

    valid = timestep_pad_mask[:, :, None, None] & action_pad_mask
    weights = jnp.ones_like(actions, dtype=jnp.float32)
    if FLAGS.binary_gripper:
        dim_weights = jnp.ones((ACTION_DIM,), dtype=jnp.float32).at[GRIPPER_INDEX].set(
            FLAGS.gripper_loss_weight
        )
        weights = weights * dim_weights
        close_multiplier = jnp.where(
            actions[..., GRIPPER_INDEX] <= 0.5, FLAGS.close_gripper_loss_weight, 1.0
        )
        weights = weights.at[..., GRIPPER_INDEX].multiply(close_multiplier)
    weights = rearrange(
        weights * valid.astype(jnp.float32), "b w h a -> b w (h a)"
    )[None]
    err = jnp.square(pred_eps - noise)
    loss = jnp.sum(err * weights) / jnp.clip(jnp.sum(weights), a_min=1e-5)
    return loss * ACTION_DIM, {
        "loss": loss * ACTION_DIM,
        "mse": jnp.mean(err) * ACTION_DIM,
    }


def main(_):
    if FLAGS.save_dir is None:
        raise ValueError("--save_dir is required")
    if FLAGS.batch_size % jax.device_count() != 0:
        raise ValueError("Batch size must be divisible by device count")

    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")
    wandb.init(
        name=FLAGS.wandb_run_name,
        project=FLAGS.wandb_project,
        mode=None if FLAGS.wandb else "disabled",
        config={f.name: f.value for f in FLAGS.flags_by_module_dict().get(__file__, [])},
    )

    logging.info("Loading Octo checkpoint: %s", FLAGS.pretrained_path)
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    dataset_kwargs = {
        "image_obs_keys": {"primary": "image_primary", "wrist": "image_wrist"},
        "proprio_obs_key": "state",
        "language_key": "language_instruction",
        "action_normalization_mask": [True] * (ACTION_DIM - 1) + [not FLAGS.binary_gripper],
    }
    traj_kwargs = {"window_size": WINDOW_SIZE, "action_horizon": ACTION_HORIZON}
    train_frames = {
        "resize_size": IMAGE_RESIZE,
        "image_augment_kwargs": IMAGE_AUGMENT,
    }
    val_frames = {"resize_size": IMAGE_RESIZE}

    train_dataset = make_interleaved_dataset(
        dataset_kwargs_list=[
            {"name": FLAGS.dataset, "data_dir": FLAGS.data_dir, **dataset_kwargs}
        ],
        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
        traj_transform_kwargs=traj_kwargs,
        frame_transform_kwargs=train_frames,
        train=True,
        traj_transform_threads=FLAGS.data_threads,
        traj_read_threads=FLAGS.data_threads,
    )
    val_dataset = make_interleaved_dataset(
        dataset_kwargs_list=[
            {"name": FLAGS.dataset, "data_dir": FLAGS.data_dir, **dataset_kwargs}
        ],
        shuffle_buffer_size=500,
        traj_transform_kwargs=traj_kwargs,
        frame_transform_kwargs=val_frames,
        train=False,
        traj_transform_threads=2,
        traj_read_threads=2,
    )

    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    def new_train_iter():
        return map(process_batch, train_dataset.repeat().batch(FLAGS.batch_size).iterator())

    def new_val_iter():
        return map(process_batch, val_dataset.repeat().batch(FLAGS.batch_size).iterator())

    train_iter = new_train_iter()
    example_batch = next(train_iter)
    if example_batch["action"].shape[-2:] != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"Unexpected action shape: {example_batch['action'].shape}")

    config = copy.deepcopy(pretrained_model.config)
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-180.0,
        high=180.0,
        obs_keys=["proprio"],
    )
    config["model"]["heads"]["action"]["kwargs"]["action_dim"] = ACTION_DIM
    config["model"]["heads"]["action"]["kwargs"]["action_horizon"] = ACTION_HORIZON

    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=train_dataset.dataset_statistics,
    )
    model = model.replace(params=merge_params(model.params, pretrained_model.params))
    del pretrained_model

    decay_steps = max(1, FLAGS.num_train_steps - FLAGS.warmup_steps)
    alpha = FLAGS.end_learning_rate / FLAGS.learning_rate
    learning_rate = optax.join_schedules(
        [
            optax.linear_schedule(0.0, FLAGS.learning_rate, FLAGS.warmup_steps),
            optax.cosine_decay_schedule(FLAGS.learning_rate, decay_steps, alpha=alpha),
        ],
        [FLAGS.warmup_steps],
    )
    backbone_alpha = (
        FLAGS.end_learning_rate / FLAGS.backbone_learning_rate
        if FLAGS.backbone_learning_rate > 0
        else 0.0
    )
    backbone_learning_rate = optax.join_schedules(
        [
            optax.linear_schedule(0.0, FLAGS.backbone_learning_rate, FLAGS.warmup_steps),
            optax.cosine_decay_schedule(
                FLAGS.backbone_learning_rate, decay_steps, alpha=backbone_alpha
            ),
        ],
        [FLAGS.warmup_steps],
    )

    def adamw(lr):
        return optax.chain(
            optax.clip_by_global_norm(FLAGS.clip_gradient),
            optax.adamw(lr, weight_decay=FLAGS.weight_decay),
        )

    # The from-scratch head needs 3e-4; the pretrained transformer needs far less.
    def label_group(path, _):
        name = ".".join(path)
        return "backbone" if name.startswith("octo_transformer") and "proprio" not in name else "head"

    tx = optax.multi_transform(
        {"head": adamw(learning_rate), "backbone": adamw(backbone_learning_rate)},
        flax.traverse_util.path_aware_map(label_group, model.params),
    )
    if FLAGS.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=FLAGS.grad_accum_steps)

    # Octo-Base config freezes only the T5 hf_model. No SmallStem patterns are added.
    frozen_keys = list(model.config.get("optimizer", {}).get("frozen_keys", []))
    frozen_keys.extend([
        "octo_transformer.observation_tokenizers_primary.SmallStem16_0.*",
        "octo_transformer.observation_tokenizers_wrist.SmallStem16_0.*",
    ])
    if FLAGS.freeze_backbone:
        # Everything under octo_transformer except the (from-scratch) proprio path.
        frozen_keys.extend(
            f"octo_transformer.{key}.*"
            for key in model.params["octo_transformer"]
            if "proprio" not in key
        )
    logging.info("Final frozen_keys: %s", frozen_keys)
    logging.info(
        "LR: head=%.1e backbone=%.1e | effective batch=%d",
        FLAGS.learning_rate,
        FLAGS.backbone_learning_rate,
        FLAGS.batch_size * max(1, FLAGS.grad_accum_steps),
    )
    logging.info(
        "Trainable: %s",
        "proprio tokenizer + action head only (backbone frozen)"
        if FLAGS.freeze_backbone
        else "Octo transformer + proprio + action head; SmallStem vision encoders frozen",
    )
    tx = freeze_weights(tx, model.params, frozen_keys)
    state = TrainState.create(rng=jax.random.PRNGKey(1234), model=model, tx=tx)

    def loss_fn(params, batch, rng, train=True):
        bound = model.module.bind({"params": params}, rngs={"dropout": rng})
        embeddings = bound.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        return weighted_action_loss(
            bound.heads["action"],
            embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )

    @jax.jit
    def train_step(train_state, batch):
        rng, dropout_rng = jax.random.split(train_state.rng)
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            train_state.model.params, batch, dropout_rng, train=True
        )
        return train_state.apply_gradients(grads=grads, rng=rng), metrics

    @jax.jit
    def val_step(train_state, batch):
        rng, dropout_rng = jax.random.split(train_state.rng)
        loss, _ = loss_fn(train_state.model.params, batch, dropout_rng, train=False)
        return loss

    def eval_val_loss(train_state):
        val_iter = new_val_iter()  # fresh iterator each time: same shuffle-stage leak as train
        return sum(float(val_step(train_state, next(val_iter))) for _ in range(FLAGS.val_batches)) / FLAGS.val_batches

    best_val = float("inf")
    best_step = -1
    patience = 0
    best_dir = os.path.join(FLAGS.save_dir, "best_val")
    final_step = FLAGS.num_train_steps - 1

    logging.info("Starting fine-tuning for %d steps", FLAGS.num_train_steps)
    for i in tqdm.tqdm(range(FLAGS.num_train_steps), dynamic_ncols=True):
        if FLAGS.reset_iter_every > 0 and i > 0 and i % FLAGS.reset_iter_every == 0:
            train_iter = new_train_iter()
        state, metrics = train_step(state, next(train_iter))
        if (i + 1) % FLAGS.log_interval == 0:
            wandb.log(
                flax.traverse_util.flatten_dict(
                    {"training": jax.device_get(metrics)}, sep="/"
                ),
                step=i,
            )
        if (i + 1) % FLAGS.val_interval == 0:
            val_loss = eval_val_loss(state)
            wandb.log({"validation/loss": val_loss}, step=i)
            logging.info("[Step %d] val_loss=%.6f (best %.6f)", i + 1, val_loss, best_val)
            if val_loss < best_val:
                best_val, best_step, patience = val_loss, i, 0
                state.model.save_pretrained(step=i, checkpoint_path=best_dir)
                logging.info("[Step %d] New best val_loss=%.6f", i + 1, val_loss)
            else:
                patience += FLAGS.val_interval
            if FLAGS.early_stop_patience > 0 and patience >= FLAGS.early_stop_patience:
                final_step = i
                logging.info("Early stopping at step %d", i + 1)
                break
        if (i + 1) % FLAGS.save_interval == 0:
            state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)

    state.model.save_pretrained(step=final_step, checkpoint_path=FLAGS.save_dir)
    wandb.log(
        {
            "final/best_val_loss": best_val,
            "final/best_val_step": best_step,
            "final/total_steps": final_step + 1,
        },
        step=final_step,
    )
    logging.info("Best val_loss=%.6f at step %d", best_val, best_step + 1)


if __name__ == "__main__":
    app.run(main)
