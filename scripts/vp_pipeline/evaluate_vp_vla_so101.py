#!/usr/bin/env python3
"""Evaluate a VP-VLA checkpoint on the fixed SO101 validation split."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from train_vp_vla_so101 import (
    build_dataset,
    build_episode_split,
    compute_grounding_loss,
    load_compatible_checkpoint,
)

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "VP-VLA"))
from starVLA.dataloader.visual_prompt_datasets import collate_fn
from starVLA.model.framework import build_framework


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_batches", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    _, val_episodes, _ = build_episode_split(
        cfg.datasets.vla_data, int(cfg.trainer.val_episodes_per_group), int(cfg.trainer.val_seed)
    )
    _, dataset = build_dataset(cfg, "val", val_episodes)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = build_framework(cfg).cuda().eval()
    load_compatible_checkpoint(model, args.checkpoint)
    preds, targets = [], []
    grounding_losses = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= args.num_batches:
                break
            preds.append(model.predict_action(batch)["normalized_actions"])
            targets.append(np.asarray([sample["action"][-16:] for sample in batch]))
            grounding_losses.append(float(compute_grounding_loss(model, batch, cfg, "cuda")))

    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    pos_pred, pos_target = pred[..., :3], target[..., :3]
    pos_norm = np.linalg.norm(pos_target, axis=-1)
    cosine = np.sum(pos_pred * pos_target, axis=-1) / (
        np.linalg.norm(pos_pred, axis=-1) * pos_norm + 1e-8
    )
    moving = pos_norm > 0.05
    pred_close, target_close = pred[..., 6] <= 0.5, target[..., 6] <= 0.5
    result = {
        "checkpoint": args.checkpoint,
        "samples": int(len(pred)),
        "normalized_action_mae": float(np.abs(pred - target).mean()),
        "normalized_continuous_mae": float(np.abs(pred[..., :6] - target[..., :6]).mean()),
        "gripper_accuracy": float((pred_close == target_close).mean()),
        "translation_cosine_moving": float(cosine[moving].mean()),
        "translation_positive_direction_rate": float((cosine[moving] > 0).mean()),
        "predicted_translation_norm": float(np.linalg.norm(pos_pred, axis=-1).mean()),
        "target_translation_norm": float(pos_norm.mean()),
        "grounding_loss": float(np.mean(grounding_losses)),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
