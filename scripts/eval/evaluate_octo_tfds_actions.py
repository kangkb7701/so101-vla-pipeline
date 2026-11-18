#!/usr/bin/env python3
"""Compare a fine-tuned Octo checkpoint against TFDS ground-truth actions.

This is an offline sanity check for the policy itself. It uses the same Octo
dataset pipeline as fine-tuning, then asks the checkpoint to predict an action
chunk from those training/validation observations.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from octo.data.dataset import make_interleaved_dataset
from octo.model.octo_model import OctoModel
from octo.utils.train_utils import process_text


SO101_DATASET = "so101_pick_place_vp_sam3"
ACTION_DIM = 7
ACTION_HORIZON = 16
WINDOW_SIZE = 2
IMAGE_RESIZE = {
    "primary": (256, 256),
    "wrist": (128, 128),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Octo action sanity check on VP-SAM3 TFDS samples."
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/home/aivlab/kkb_capstone/checkpoints/vp_sam3_pick_place_h16/best_val",
        help="Octo checkpoint directory.",
    )
    parser.add_argument("--step", type=int, default=899, help="Checkpoint step. Use -1 for latest/default.")
    parser.add_argument("--data_dir", default="/home/aivlab/tensorflow_datasets")
    parser.add_argument("--dataset", default=SO101_DATASET)
    parser.add_argument("--action_horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument(
        "--binary_gripper",
        action="store_true",
        help="Use action_normalization_mask=[True]*6+[False] for binary gripper datasets.",
    )
    parser.add_argument("--action_dim", type=int, default=ACTION_DIM, help="7 for delta-EE datasets, 6 for joint datasets.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Diffusion samples to average per observation. The head samples from the "
        "learned action distribution, so on a noisy target a single sample is worse than "
        "the conditional mean; averaging approximates it.",
    )
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--samples_to_print", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_npz",
        default="outputs/octo_tfds_action_eval/latest_eval.npz",
        help="Where to save pred/gt arrays. Empty string disables saving.",
    )
    return parser.parse_args()


def tree_to_numpy(batch):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), batch)


def make_dataset(args: argparse.Namespace):
    dataset_kwargs = dict(
        image_obs_keys={
            "primary": "image_primary",
            "wrist": "image_wrist",
        },
        proprio_obs_key="state",
        language_key="language_instruction",
    )
    if args.binary_gripper:
        dataset_kwargs["action_normalization_mask"] = [True] * (args.action_dim - 1) + [False]
    traj_transform_kwargs = dict(
        window_size=WINDOW_SIZE,
        action_horizon=args.action_horizon,
    )
    frame_transform_kwargs = dict(resize_size=IMAGE_RESIZE)
    ds = make_interleaved_dataset(
        dataset_kwargs_list=[
            dict(name=args.dataset, data_dir=args.data_dir, **dataset_kwargs),
        ],
        shuffle_buffer_size=1000 if args.split == "train" else 1,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        train=args.split == "train",
    )
    return ds


def print_action_stats(name: str, arr: np.ndarray) -> None:
    flat = arr.reshape(-1, arr.shape[-1])
    print(f"\n{name}: shape={arr.shape}")
    print("  mean:", np.round(flat.mean(axis=0), 5).tolist())
    print("  std :", np.round(flat.std(axis=0), 5).tolist())
    print("  min :", np.round(flat.min(axis=0), 5).tolist())
    print("  max :", np.round(flat.max(axis=0), 5).tolist())
    print("  p01 :", np.round(np.quantile(flat, 0.01, axis=0), 5).tolist())
    print("  p50 :", np.round(np.quantile(flat, 0.50, axis=0), 5).tolist())
    print("  p99 :", np.round(np.quantile(flat, 0.99, axis=0), 5).tolist())


def gripper_event_mask(actions: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    return np.abs(actions[..., -1]) >= threshold


def binary_gripper_close_mask(actions: np.ndarray) -> np.ndarray:
    return actions[..., -1] <= 0.5


def main() -> None:
    args = parse_args()
    tf.config.set_visible_devices([], "GPU")

    step = None if args.step < 0 else args.step
    print(f"Loading Octo checkpoint: {args.checkpoint_path} step={step}")
    model = OctoModel.load_pretrained(args.checkpoint_path, step=step)
    dataset_key = list(model.dataset_statistics.keys())[0]
    action_stats = model.dataset_statistics[dataset_key]["action"]
    action_dim = int(np.asarray(action_stats["mean"]).shape[-1])
    gripper_idx = action_dim - 1
    print(f"Using dataset statistics key: {dataset_key}")

    ds = make_dataset(args)
    iterator = ds.repeat().batch(args.batch_size).iterator()

    rng = jax.random.PRNGKey(args.seed)
    all_pred = []
    all_gt = []

    for batch_idx in range(args.num_batches):
        batch = next(iterator)
        batch = process_text(batch, model.text_processor)
        batch.pop("dataset_name", None)

        samples = []
        for _ in range(args.num_samples):
            rng, sample_rng = jax.random.split(rng)
            samples.append(
                np.asarray(
                    jax.device_get(
                        model.sample_actions(
                            batch["observation"], batch["task"], rng=sample_rng
                        )
                    ),
                    dtype=np.float32,
                )
            )
        pred_norm_np = np.mean(samples, axis=0)
        gt_norm_np = np.asarray(jax.device_get(batch["action"]), dtype=np.float32)
        if gt_norm_np.ndim == 4:
            # Octo's windowed dataloader returns [B, window, horizon, D]. The
            # policy predicts the action chunk for the latest observation.
            gt_norm_np = gt_norm_np[:, -1]
        if pred_norm_np.shape != gt_norm_np.shape:
            raise ValueError(f"pred/gt shape mismatch: pred={pred_norm_np.shape}, gt={gt_norm_np.shape}")

        mean = np.asarray(action_stats["mean"], dtype=np.float32)
        std = np.asarray(action_stats["std"], dtype=np.float32)
        mask = np.asarray(action_stats.get("mask", np.ones_like(mean, dtype=bool)), dtype=bool)
        pred_raw_np = np.where(mask, pred_norm_np * std + mean, pred_norm_np)
        gt_raw_np = np.where(mask, gt_norm_np * std + mean, gt_norm_np)

        all_pred.append(pred_raw_np)
        all_gt.append(gt_raw_np)
        print(f"batch {batch_idx}: pred={pred_raw_np.shape} gt={gt_raw_np.shape}")

    pred = np.concatenate(all_pred, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    err = pred - gt
    abs_err = np.abs(err)

    print_action_stats("GT actions", gt)
    print_action_stats("Pred actions", pred)
    print_action_stats("Pred-GT error", err)

    print("\nMAE per dim over full horizon:")
    print(np.round(abs_err.reshape(-1, action_dim).mean(axis=0), 5).tolist())
    print("RMSE per dim over full horizon:")
    print(np.round(np.sqrt((err.reshape(-1, action_dim) ** 2).mean(axis=0)), 5).tolist())

    first_mae = abs_err[:, 0, :].mean(axis=0)
    print("\nMAE per dim at horizon[0]:")
    print(np.round(first_mae, 5).tolist())

    if args.binary_gripper:
        gt_close = binary_gripper_close_mask(gt)
        pred_close = binary_gripper_close_mask(pred)
        print("\nBinary gripper diagnostic | close := gripper_open <= 0.5")
        print(f"  gt_close_rate={float(gt_close.mean()):.4f}")
        print(f"  pred_close_rate={float(pred_close.mean()):.4f}")
        print(f"  close_agreement={float((gt_close == pred_close).mean()):.4f}")
    else:
        gt_event = gripper_event_mask(gt)
        pred_event = gripper_event_mask(pred)
        gt_event_rate = float(gt_event.mean())
        pred_event_rate = float(pred_event.mean())
        event_agree = float((gt_event == pred_event).mean())
        print("\nGripper event diagnostic | abs(delta_gripper) >= 1.0 deg")
        print(f"  gt_event_rate={gt_event_rate:.4f}")
        print(f"  pred_event_rate={pred_event_rate:.4f}")
        print(f"  event_agreement={event_agree:.4f}")

    n_print = min(args.samples_to_print, pred.shape[0])
    print(f"\nFirst {n_print} samples, horizon[0] GT vs Pred:")
    for i in range(n_print):
        gt0 = gt[i, 0]
        pr0 = pred[i, 0]
        print(
            f"[{i:02d}] "
            f"gt={np.round(gt0[:gripper_idx], 4).tolist()} "
            f"pred={np.round(pr0[:gripper_idx], 4).tolist()} | "
            f"gt_grip={gt0[gripper_idx]:+.3f} pred_grip={pr0[gripper_idx]:+.3f}"
        )

    if args.save_npz:
        out = Path(args.save_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, pred=pred, gt=gt, error=err)
        print(f"\nSaved arrays: {out}")


if __name__ == "__main__":
    main()
