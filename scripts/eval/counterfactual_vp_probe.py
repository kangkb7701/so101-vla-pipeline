#!/usr/bin/env python3
"""Counterfactual visual-prompt probe for Octo checkpoints.

The test keeps the TFDS observation, state, and language fixed, then draws
different synthetic crosshairs on image_primary and compares predicted actions.
If the policy is using visual prompts, large prompt moves should produce
measurable action changes.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from octo.data.dataset import make_interleaved_dataset
from octo.model.octo_model import OctoModel
from octo.utils.train_utils import process_text


ACTION_DIM = 7
WINDOW_SIZE = 2
IMAGE_RESIZE = {
    "primary": (256, 256),
    "wrist": (128, 128),
}

VARIANT_POINTS: Dict[str, Tuple[float, float] | None] = {
    "original": None,
    "center": (0.50, 0.50),
    "left": (0.25, 0.50),
    "right": (0.75, 0.50),
    "upper": (0.50, 0.25),
    "lower": (0.50, 0.75),
    "upper_left": (0.25, 0.25),
    "lower_right": (0.75, 0.75),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe whether Octo reacts to visual prompt position changes.")
    parser.add_argument(
        "--checkpoint_path",
        default="/home/aivlab/kkb_capstone/checkpoints/vp_sam3_pick_place_h4_binary_gripper_noep60_top_unfrozen_gw/best_val",
    )
    parser.add_argument("--step", type=int, default=299, help="Checkpoint step. Use -1 for latest/default.")
    parser.add_argument("--data_dir", default="/home/aivlab/tensorflow_datasets")
    parser.add_argument("--dataset", default="so101_pick_place_vp_sam3_binary_gripper_noep60")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draw_all_timesteps", action="store_true", help="Draw prompt on every history frame.")
    parser.add_argument("--save_dir", default="outputs/counterfactual_vp_probe")
    return parser.parse_args()


def make_dataset(args: argparse.Namespace):
    dataset_kwargs = dict(
        image_obs_keys={
            "primary": "image_primary",
            "wrist": "image_wrist",
        },
        proprio_obs_key="state",
        language_key="language_instruction",
        action_normalization_mask=[True, True, True, True, True, True, False],
    )
    return make_interleaved_dataset(
        dataset_kwargs_list=[
            dict(name=args.dataset, data_dir=args.data_dir, **dataset_kwargs),
        ],
        shuffle_buffer_size=1000 if args.split == "train" else 1,
        traj_transform_kwargs=dict(window_size=WINDOW_SIZE, action_horizon=args.action_horizon),
        frame_transform_kwargs=dict(resize_size=IMAGE_RESIZE),
        train=args.split == "train",
    )


def draw_crosshair_np(image: np.ndarray, point: Tuple[float, float]) -> np.ndarray:
    """Draw a VP-VLA-style RGB crosshair at normalized image coordinates."""
    img = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x = int(round(point[0] * (w - 1)))
    y = int(round(point[1] * (h - 1)))
    line_length = 14
    gap = 7
    thickness = 3
    green = (0, 255, 0)
    red = (255, 0, 0)
    draw.line((x - gap - line_length, y, x - gap, y), fill=green, width=thickness)
    draw.line((x + gap, y, x + gap + line_length, y), fill=green, width=thickness)
    draw.line((x, y - gap - line_length, x, y - gap), fill=green, width=thickness)
    draw.line((x, y + gap, x, y + gap + line_length), fill=green, width=thickness)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=red)
    return np.asarray(img, dtype=np.uint8)


def clone_batch_np(batch):
    return jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), batch)


def apply_variant(batch_np, point: Tuple[float, float] | None, draw_all_timesteps: bool):
    variant = copy.deepcopy(batch_np)
    if point is None:
        return variant
    imgs = variant["observation"]["image_primary"]
    # Shape: [B, window, H, W, C]
    timestep_indices = range(imgs.shape[1]) if draw_all_timesteps else [imgs.shape[1] - 1]
    for b in range(imgs.shape[0]):
        for t in timestep_indices:
            imgs[b, t] = draw_crosshair_np(imgs[b, t], point)
    variant["observation"]["image_primary"] = imgs
    return variant


def unnormalize(pred_norm: np.ndarray, action_stats: dict) -> np.ndarray:
    mean = np.asarray(action_stats["mean"], dtype=np.float32)
    std = np.asarray(action_stats["std"], dtype=np.float32)
    mask = np.asarray(action_stats.get("mask", np.ones_like(mean, dtype=bool)), dtype=bool)
    return np.where(mask, pred_norm * std + mean, pred_norm)


def pos_norm_cm(actions: np.ndarray) -> np.ndarray:
    return np.linalg.norm(actions[..., :3], axis=-1) * 100.0


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return dot / np.maximum(denom, eps)


def save_preview(batch_np, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    base = batch_np["observation"]["image_primary"][0, -1]
    for name, point in VARIANT_POINTS.items():
        img = base if point is None else draw_crosshair_np(base, point)
        Image.fromarray(img).save(save_dir / f"variant_{name}.png")


def main() -> None:
    args = parse_args()
    tf.config.set_visible_devices([], "GPU")

    step = None if args.step < 0 else args.step
    print(f"Loading Octo checkpoint: {args.checkpoint_path} step={step}")
    model = OctoModel.load_pretrained(args.checkpoint_path, step=step)
    dataset_key = list(model.dataset_statistics.keys())[0]
    action_stats = model.dataset_statistics[dataset_key]["action"]
    print(f"Using dataset statistics key: {dataset_key}")

    ds = make_dataset(args)
    iterator = ds.repeat().batch(args.batch_size).iterator()

    rng = jax.random.PRNGKey(args.seed)
    raw_preds = {name: [] for name in VARIANT_POINTS}
    raw_gts = []

    for batch_idx in range(args.num_batches):
        batch = next(iterator)
        batch = process_text(batch, model.text_processor)
        batch.pop("dataset_name", None)
        batch_np = clone_batch_np(batch)

        if batch_idx == 0:
            save_preview(batch_np, Path(args.save_dir) / "preview")

        gt = np.asarray(batch_np["action"], dtype=np.float32)
        if gt.ndim == 4:
            gt = gt[:, -1]
        raw_gts.append(unnormalize(gt, action_stats))

        rng, sample_rng = jax.random.split(rng)
        for name, point in VARIANT_POINTS.items():
            variant = apply_variant(batch_np, point, args.draw_all_timesteps)
            pred_norm = model.sample_actions(
                variant["observation"],
                variant["task"],
                rng=sample_rng,
            )
            pred_raw = unnormalize(np.asarray(jax.device_get(pred_norm), dtype=np.float32), action_stats)
            raw_preds[name].append(pred_raw)
        print(f"batch {batch_idx}: evaluated {len(VARIANT_POINTS)} variants")

    preds = {name: np.concatenate(chunks, axis=0) for name, chunks in raw_preds.items()}
    gt_raw = np.concatenate(raw_gts, axis=0)
    original = preds["original"]

    print("\n=== Counterfactual VP sensitivity ===")
    print(f"samples={original.shape[0]} horizon={original.shape[1]} draw_all_timesteps={args.draw_all_timesteps}")
    print("Action units: position deltas in cm below; raw arrays are model action units.")
    print("\nOriginal prediction pos norm cm p50/p90/p99:")
    print(np.round(np.quantile(pos_norm_cm(original), [0.5, 0.9, 0.99]), 4).tolist())
    print("GT pos norm cm p50/p90/p99:")
    print(np.round(np.quantile(pos_norm_cm(gt_raw), [0.5, 0.9, 0.99]), 4).tolist())

    print("\nVariant vs original action change:")
    rows = []
    for name, arr in preds.items():
        if name == "original":
            continue
        diff = arr - original
        first_pos_diff_cm = np.linalg.norm(diff[:, 0, :3], axis=-1) * 100.0
        full_pos_diff_cm = np.linalg.norm(diff[..., :3], axis=-1) * 100.0
        first_rot_diff = np.linalg.norm(diff[:, 0, 3:6], axis=-1)
        first_grip_diff = np.abs(diff[:, 0, 6])
        first_cos = cosine(arr[:, 0, :3], original[:, 0, :3])
        rows.append(
            (
                name,
                float(np.mean(first_pos_diff_cm)),
                float(np.quantile(first_pos_diff_cm, 0.9)),
                float(np.mean(full_pos_diff_cm)),
                float(np.mean(first_rot_diff)),
                float(np.mean(first_grip_diff)),
                float(np.mean(first_cos)),
            )
        )
    print("name         first_pos_diff_cm_mean  first_p90  full_pos_diff_cm_mean  first_rot_diff  first_grip_diff  first_pos_cos")
    for row in rows:
        print(f"{row[0]:12s} {row[1]:23.4f} {row[2]:10.4f} {row[3]:23.4f} {row[4]:15.5f} {row[5]:16.5f} {row[6]:14.4f}")

    stacked = np.stack([preds[name][..., :3] for name in VARIANT_POINTS if name != "original"], axis=0)
    spread_cm = np.linalg.norm(np.std(stacked, axis=0), axis=-1) * 100.0
    print("\nInter-variant positional spread cm:")
    print("  p50/p90/p99:", np.round(np.quantile(spread_cm, [0.5, 0.9, 0.99]), 4).tolist())

    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "counterfactual_preds.npz",
        gt=gt_raw,
        **{f"pred_{name}": arr for name, arr in preds.items()},
    )
    print(f"\nSaved results: {out / 'counterfactual_preds.npz'}")
    print(f"Saved preview images: {out / 'preview'}")


if __name__ == "__main__":
    main()
