"""
SO-101 Octo fine-tuning on VP-SAM3 pick-and-place data with auxiliary grounding.

Compared with full_finetuning_vp_sam3_pick_place_h4_binary_gripper.py:
  - Uses a TFDS builder that includes observation.vp_coord/vp_valid/vp_phase.
  - Adds a lightweight GroundingHead on Octo's action readout tokens.
  - Optimizes: total = weighted diffusion action loss + lambda * grounding MSE.
  - Saves inference-compatible checkpoints with the auxiliary head stripped.

Grounding target layout:
  vp_coord = [point_x, point_y, box_x1, box_y1, box_x2, box_y2]
  vp_valid = [point_valid, box_valid]
All coordinates are normalized to [0, 1] in the original top-camera frame.
"""

from __future__ import annotations

from absl import app, flags, logging
import copy
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import dlimp as dl
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import wandb
from einops import rearrange

REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

from octo.data import obs_transforms, traj_transforms
from octo.data.dataset import apply_frame_transforms, apply_trajectory_transforms
from octo.data.utils.data_utils import (
    NormalizationType,
    get_dataset_statistics,
    normalize_action_and_proprio,
    sample_match_keys_uniform,
    tree_map,
)
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
flags.DEFINE_string(
    "dataset",
    "so101_pick_place_vp_sam3_binary_gripper_noep60_grounding",
    "TFDS dataset name with vp_coord/vp_valid labels.",
)
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
flags.DEFINE_string("wandb_run_name", "vp_sam3_pick_place_h4_binary_gripper_grounding", "wandb run name.")
flags.DEFINE_bool("freeze_primary_vision", False, "Freeze top/primary image tokenizer and ViT.")
flags.DEFINE_bool("freeze_wrist_vision", True, "Freeze wrist image tokenizer and ViT.")
flags.DEFINE_float("gripper_loss_weight", 2.0, "Extra diffusion loss weight for gripper action dimension.")
flags.DEFINE_float("close_gripper_loss_weight", 2.0, "Additional gripper-dim weight when target gripper action is close (0).")
flags.DEFINE_float("grounding_loss_weight", 1.0, "Weight for auxiliary visual grounding MSE.")
flags.DEFINE_integer("grounding_hidden_dim", 256, "Hidden width for the auxiliary grounding MLP.")


SO101_ACTION_DIM = 7
SO101_ACTION_HORIZON = 4
WINDOW_SIZE = 2
GROUNDING_COORD_DIM = 6
GROUNDING_OBS_KEYS = ("vp_coord", "vp_valid", "vp_phase")

IMAGE_RESIZE = {
    "primary": (256, 256),
    "wrist": (128, 128),
}

EXTRA_FROZEN_PATTERNS = []


def _require_flags():
    if FLAGS.save_dir is None:
        raise ValueError("--save_dir is required")


def make_dataset_from_rlds_with_grounding(
    name: str,
    data_dir: str,
    *,
    train: bool,
    shuffle: bool = True,
    image_obs_keys: Mapping[str, Optional[str]] = {},
    depth_obs_keys: Mapping[str, Optional[str]] = {},
    proprio_obs_key: Optional[str] = None,
    language_key: Optional[str] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    force_recompute_dataset_statistics: bool = False,
    action_normalization_mask: Optional[Sequence[bool]] = None,
    skip_norm: bool = False,
    ignore_errors: bool = False,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
    """Octo RLDS loader variant that preserves vp_coord/vp_valid/vp_phase."""

    def restructure(traj):
        if "observation" not in traj or "action" not in traj:
            raise ValueError("Trajectory must contain observation and action")

        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

        if proprio_obs_key is not None:
            new_obs["proprio"] = tf.cast(old_obs[proprio_obs_key], tf.float32)

        for key in GROUNDING_OBS_KEYS:
            if key not in old_obs:
                raise KeyError(
                    f"Dataset {name} does not contain observation.{key}. "
                    "Build it with so101_vp_sam3_pick_place_grounding_dataset_builder.py."
                )
            new_obs[key] = tf.cast(old_obs[key], tf.float32)

        new_obs["timestep"] = tf.range(traj_len)

        task = {}
        if language_key is not None:
            task["language_instruction"] = sample_match_keys_uniform(traj, language_key)
            if task["language_instruction"].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {task['language_instruction'].dtype}, "
                    "but it must be tf.string."
                )

        return {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }

    def is_nonzero_length(traj):
        return tf.shape(traj["action"])[0] > 0

    builder = tfds.builder(name, data_dir=data_dir)

    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False)
        if ignore_errors:
            full_dataset = full_dataset.ignore_errors()
        full_dataset = full_dataset.traj_map(restructure).filter(is_nonzero_length)
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(proprio_obs_key),
                "vp_grounding_loader_v1",
            ),
            save_dir=builder.data_dir,
            force_recompute=force_recompute_dataset_statistics,
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)

    if action_normalization_mask is not None:
        if (
            len(action_normalization_mask)
            != dataset_statistics["action"]["mean"].shape[-1]
        ):
            raise ValueError(
                f"Length of action_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )
    if ignore_errors:
        dataset = dataset.ignore_errors()
    dataset = dataset.traj_map(restructure, num_parallel_calls).filter(is_nonzero_length)

    if not skip_norm:
        dataset = dataset.traj_map(
            partial(
                normalize_action_and_proprio,
                metadata=dataset_statistics,
                normalization_type=action_proprio_normalization_type,
            ),
            num_parallel_calls,
        )
    else:
        logging.warning("Dataset normalization is disabled.")

    return dataset, dataset_statistics


def make_grounding_dataset(
    dataset_kwargs: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict,
    frame_transform_kwargs: dict,
    dataset_statistics: Optional[dict] = None,
) -> dl.DLataset:
    dataset, stats = make_dataset_from_rlds_with_grounding(
        **dataset_kwargs,
        train=train,
        dataset_statistics=dataset_statistics,
    )
    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = dataset.flatten()
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)
    dataset = dataset.with_ram_budget(1)
    dataset.dataset_statistics = {dataset_kwargs["name"]: stats}
    return dataset


def _weighted_diffusion_action_loss(
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
        "action_loss": loss,
        "action_mse": mse * SO101_ACTION_DIM,
        "gripper_mse": gripper_mse,
        "gripper_loss_weight": jnp.asarray(FLAGS.gripper_loss_weight, dtype=jnp.float32),
        "close_gripper_loss_weight": jnp.asarray(FLAGS.close_gripper_loss_weight, dtype=jnp.float32),
    }


def _grounding_loss(grounding_head, transformer_embeddings, observation, train=True):
    pred = grounding_head(transformer_embeddings, train=train)
    target = observation["vp_coord"]
    valid = observation["vp_valid"]
    timestep_mask = observation["timestep_pad_mask"].astype(jnp.float32)

    point_mask = valid[..., 0:1].repeat(2, axis=-1)
    box_mask = valid[..., 1:2].repeat(4, axis=-1)
    coord_mask = jnp.concatenate([point_mask, box_mask], axis=-1)
    coord_mask = coord_mask * timestep_mask[..., None]

    err = jnp.square(pred - target)
    denom = jnp.clip(jnp.sum(coord_mask), a_min=1e-5)
    loss = jnp.sum(err * coord_mask) / denom

    point_denom = jnp.clip(jnp.sum(coord_mask[..., :2]), a_min=1e-5)
    box_denom = jnp.clip(jnp.sum(coord_mask[..., 2:]), a_min=1e-5)
    point_mse = jnp.sum(err[..., :2] * coord_mask[..., :2]) / point_denom
    box_mse = jnp.sum(err[..., 2:] * coord_mask[..., 2:]) / box_denom

    point_valid_rate = jnp.mean(valid[..., 0] * timestep_mask)
    box_valid_rate = jnp.mean(valid[..., 1] * timestep_mask)

    return loss, {
        "grounding_loss": loss,
        "grounding_point_mse": point_mse,
        "grounding_box_mse": box_mse,
        "grounding_point_valid_rate": point_valid_rate,
        "grounding_box_valid_rate": box_valid_rate,
        "grounding_loss_weight": jnp.asarray(FLAGS.grounding_loss_weight, dtype=jnp.float32),
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


def _add_grounding_head(config):
    action_readout = config["model"]["heads"]["action"]["kwargs"].get(
        "readout_key", "readout_action"
    )
    config["model"]["heads"]["grounding"] = ModuleSpec.create(
        "datasets.octo_grounding_head:GroundingHead",
        readout_key=action_readout,
        hidden_dim=FLAGS.grounding_hidden_dim,
        output_dim=GROUNDING_COORD_DIM,
    )
    logging.info(
        "Grounding head added on readout=%s hidden_dim=%d",
        action_readout,
        FLAGS.grounding_hidden_dim,
    )


def _strip_grounding_for_save(model: OctoModel) -> OctoModel:
    """Return an inference-compatible model without grounding head/labels."""
    config = copy.deepcopy(model.config)
    config["model"]["heads"].pop("grounding", None)

    params = flax.core.unfreeze(model.params)
    params.pop("heads_grounding", None)
    params = flax.core.freeze(params)

    example_batch = copy.deepcopy(model.example_batch)
    obs = example_batch["observation"]
    for key in GROUNDING_OBS_KEYS:
        obs.pop(key, None)
    pad_mask_dict = obs.get("pad_mask_dict", None)
    if pad_mask_dict is not None:
        for key in GROUNDING_OBS_KEYS:
            pad_mask_dict.pop(key, None)

    return model.replace(config=config, params=params, example_batch=example_batch)


def _save_inference_checkpoint(model: OctoModel, step: int, checkpoint_path: str):
    stripped = _strip_grounding_for_save(model)
    stripped.save_pretrained(step=step, checkpoint_path=checkpoint_path)


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
        name=FLAGS.dataset,
        data_dir=FLAGS.data_dir,
        image_obs_keys={
            "primary": "image_primary",
            "wrist": "image_wrist",
        },
        proprio_obs_key="state",
        language_key="language_instruction",
        action_normalization_mask=[True, True, True, True, True, True, False],
    )
    traj_transform_kwargs = dict(
        window_size=WINDOW_SIZE,
        action_horizon=SO101_ACTION_HORIZON,
    )
    frame_transform_kwargs = dict(resize_size=IMAGE_RESIZE)

    logging.info("Loading train split from TFDS dataset: %s", FLAGS.dataset)
    train_dataset = make_grounding_dataset(
        dataset_kwargs,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        train=True,
    )
    train_stats = train_dataset.dataset_statistics[FLAGS.dataset]
    train_data_iter = train_dataset.repeat().batch(FLAGS.batch_size).iterator()

    logging.info("Loading val split from TFDS dataset: %s", FLAGS.dataset)
    val_dataset = make_grounding_dataset(
        dataset_kwargs,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        dataset_statistics=train_stats,
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
    logging.info("Example vp_coord shape: %s", example_batch["observation"]["vp_coord"].shape)
    logging.info("Example vp_valid shape: %s", example_batch["observation"]["vp_valid"].shape)
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
    _add_grounding_head(config)

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
        "Trainable: transformer + proprio tokenizer + action head + grounding head + "
        "%s primary vision; frozen wrist vision=%s",
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
        grounding_loss, grounding_metrics = _grounding_loss(
            bound_module.heads["grounding"],
            transformer_embeddings,
            batch["observation"],
            train=train,
        )
        total_loss = action_loss + FLAGS.grounding_loss_weight * grounding_loss
        metrics = {
            "loss": total_loss,
            **action_metrics,
            **grounding_metrics,
        }
        return total_loss, metrics

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

    def eval_val_metrics(state, n_batches=FLAGS.val_batches):
        losses = []
        metric_acc = {}
        for _ in range(n_batches):
            batch = next(val_data_iter)
            loss, info = val_step(state, batch)
            info = jax.device_get(info)
            losses.append(float(loss))
            for key, value in info.items():
                metric_acc.setdefault(key, []).append(float(value))
        metrics = {key: sum(values) / len(values) for key, values in metric_acc.items()}
        metrics["loss"] = sum(losses) / len(losses)
        return metrics

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
            val_metrics = eval_val_metrics(train_state)
            val_loss = val_metrics["loss"]
            wandb.log(
                flax.traverse_util.flatten_dict({"validation": val_metrics}, sep="/"),
                step=i,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_step = i
                _save_inference_checkpoint(train_state.model, i, best_ckpt_dir)
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
            _save_inference_checkpoint(train_state.model, i, FLAGS.save_dir)

    final_step = i if early_stopped else FLAGS.num_train_steps - 1
    _save_inference_checkpoint(train_state.model, final_step, FLAGS.save_dir)
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
            "final/grounding_loss_weight": FLAGS.grounding_loss_weight,
        },
        step=final_step,
    )


if __name__ == "__main__":
    app.run(main)
