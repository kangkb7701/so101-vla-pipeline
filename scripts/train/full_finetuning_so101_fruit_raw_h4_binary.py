"""Fine-tune Octo-Base 1.5 on raw SO-101 top+side pick-and-place data.

The training structure follows checkpoint7, but freezes only the pretrained T5
language encoder and both SmallStem vision encoders. The Octo transformer,
proprio tokenizer, and action head are trainable.
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
flags.DEFINE_string("dataset", "so101_fruit_raw_3basket_h4_binary", "TFDS dataset name.")
flags.DEFINE_string("save_dir", None, "Checkpoint output directory.")
flags.DEFINE_integer("batch_size", 4, "Batch size for Octo-Base on a 24GB GPU.")
flags.DEFINE_integer("num_train_steps", 8000, "Maximum fine-tuning steps.")
flags.DEFINE_integer("log_interval", 50, "Training metric interval.")
flags.DEFINE_integer("val_interval", 250, "Validation interval.")
flags.DEFINE_integer("save_interval", 1000, "Regular checkpoint interval.")
flags.DEFINE_integer("val_batches", 32, "Validation batches per evaluation.")
flags.DEFINE_float("learning_rate", 1e-6, "Peak learning rate.")
flags.DEFINE_float("end_learning_rate", 1e-7, "Final cosine-decay learning rate.")
flags.DEFINE_integer("warmup_steps", 500, "Linear warmup steps.")
flags.DEFINE_float("weight_decay", 0.01, "AdamW weight decay.")
flags.DEFINE_float("clip_gradient", 1.0, "Global gradient norm clipping.")
flags.DEFINE_integer("early_stop_patience", 0, "0 disables early stopping.")
flags.DEFINE_float("gripper_loss_weight", 2.0, "Binary gripper dimension loss weight.")
flags.DEFINE_float("close_gripper_loss_weight", 1.5, "Extra weight for close=0 targets.")
flags.DEFINE_bool("wandb", True, "Enable W&B.")
flags.DEFINE_string("wandb_project", "octo", "W&B project.")
flags.DEFINE_string("wandb_run_name", "so101_fruit_raw_3basket_base_h4_binary", "W&B run.")

ACTION_DIM = 7
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
    dim_weights = jnp.ones((ACTION_DIM,), dtype=jnp.float32).at[6].set(
        FLAGS.gripper_loss_weight
    )
    weights = weights * dim_weights
    close_multiplier = jnp.where(
        actions[..., 6] <= 0.5, FLAGS.close_gripper_loss_weight, 1.0
    )
    weights = weights.at[..., 6].multiply(close_multiplier)
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
    )

    logging.info("Loading Octo checkpoint: %s", FLAGS.pretrained_path)
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    dataset_kwargs = {
        "image_obs_keys": {"primary": "image_primary", "wrist": "image_wrist"},
        "proprio_obs_key": "state",
        "language_key": "language_instruction",
        "action_normalization_mask": [True, True, True, True, True, True, False],
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
        shuffle_buffer_size=10000,
        traj_transform_kwargs=traj_kwargs,
        frame_transform_kwargs=train_frames,
        train=True,
    )
    val_dataset = make_interleaved_dataset(
        dataset_kwargs_list=[
            {"name": FLAGS.dataset, "data_dir": FLAGS.data_dir, **dataset_kwargs}
        ],
        shuffle_buffer_size=1000,
        traj_transform_kwargs=traj_kwargs,
        frame_transform_kwargs=val_frames,
        train=False,
    )
    train_iter = train_dataset.repeat().batch(FLAGS.batch_size).iterator()
    val_iter = val_dataset.repeat().batch(FLAGS.batch_size).iterator()

    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_iter = map(process_batch, train_iter)
    val_iter = map(process_batch, val_iter)
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
    tx = optax.chain(
        optax.clip_by_global_norm(FLAGS.clip_gradient),
        optax.adamw(learning_rate, weight_decay=FLAGS.weight_decay),
    )

    # Octo-Base config freezes only the T5 hf_model. No SmallStem patterns are added.
    frozen_keys = list(model.config.get("optimizer", {}).get("frozen_keys", []))
    frozen_keys.extend([
        "octo_transformer.observation_tokenizers_primary.SmallStem16_0.*",
        "octo_transformer.observation_tokenizers_wrist.SmallStem16_0.*",
    ])
    logging.info("Final frozen_keys: %s", frozen_keys)
    logging.info(
        "Trainable: Octo transformer + proprio + action head; SmallStem vision encoders frozen"
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
        return sum(float(val_step(train_state, next(val_iter))) for _ in range(FLAGS.val_batches)) / FLAGS.val_batches

    best_val = float("inf")
    best_step = -1
    patience = 0
    best_dir = os.path.join(FLAGS.save_dir, "best_val")
    final_step = FLAGS.num_train_steps - 1

    logging.info("Starting fine-tuning for %d steps", FLAGS.num_train_steps)
    for i in tqdm.tqdm(range(FLAGS.num_train_steps), dynamic_ncols=True):
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
