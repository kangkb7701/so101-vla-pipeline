#!/usr/bin/env python3
"""Single-GPU VP-VLA action fine-tuning for the SO-101 LeRobot v3 dataset.

This trainer reuses the cloned VP-VLA model and visual-prompt dataset classes,
but avoids the repository's mandatory 8-GPU DeepSpeed launcher. The default
configuration trains the action expert plus visual-attention LoRA adapters with
an inline visual-grounding objective on one RTX 3090.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler


os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VP_VLA_ROOT = PROJECT_ROOT / "VP-VLA"
if str(VP_VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(VP_VLA_ROOT))

from starVLA.dataloader.visual_prompt_datasets import (  # noqa: E402
    VisualPromptMixtureDataset,
    collate_fn,
    make_VisualPromptSingleDataset,
)
from starVLA.model.framework import build_framework  # noqa: E402


class LoRALinear(nn.Module):
    """Minimal LoRA adapter that exports back to a regular Linear weight."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.base.requires_grad_(False)

    def forward(self, inputs):
        update = self.lora_b(self.lora_a(self.dropout(inputs)))
        return self.base(inputs) + update * self.scaling


def inject_vision_lora(
    model: nn.Module, target_pattern: str, rank: int, alpha: float, dropout: float
) -> list[str]:
    """Freeze Qwen-VL and add LoRA only to matching visual Linear layers."""
    qwen = model.qwen_vl_interface
    qwen.requires_grad_(False)
    pattern = re.compile(target_pattern)
    replacements = []
    for name, module in qwen.named_modules():
        if isinstance(module, nn.Linear) and pattern.search(name):
            parent_name, leaf_name = name.rsplit(".", 1)
            replacements.append((name, qwen.get_submodule(parent_name), leaf_name, module))
    if not replacements:
        candidates = [
            name for name, module in qwen.named_modules()
            if isinstance(module, nn.Linear) and "visual" in name
        ]
        raise ValueError(
            f"No vision LoRA targets matched {target_pattern!r}. "
            f"Example visual Linear modules: {candidates[:20]}"
        )
    for _, parent, leaf_name, module in replacements:
        setattr(parent, leaf_name, LoRALinear(module, rank, alpha, dropout))
    return [name for name, _, _, _ in replacements]


def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Build a deployment-compatible state dict with LoRA deltas merged."""
    state = model.state_dict()
    merged = {}
    consumed = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        prefix = f"{module_name}."
        base_key = prefix + "base.weight"
        base_weight = state[base_key]
        delta = module.lora_b.weight.float() @ module.lora_a.weight.float()
        merged[prefix + "weight"] = (
            base_weight.float() + delta * module.scaling
        ).to(base_weight.dtype).cpu()
        consumed.update({base_key, prefix + "lora_a.weight", prefix + "lora_b.weight"})
        bias_key = prefix + "base.bias"
        if bias_key in state:
            merged[prefix + "bias"] = state[bias_key].cpu()
            consumed.add(bias_key)
    for key, value in state.items():
        if key not in consumed:
            merged[key] = value.cpu()
    return merged


def freeze_modules(model: torch.nn.Module, module_paths: str) -> list[str]:
    frozen: list[str] = []
    for path in [x.strip() for x in module_paths.split(",") if x.strip()]:
        module = model
        for part in path.split("."):
            module = getattr(module, part)
        module.requires_grad_(False)
        frozen.append(path)
    return frozen


def load_compatible_checkpoint(model: torch.nn.Module, checkpoint_path: str | None) -> dict:
    """Load matching tensors and report incompatible keys instead of hiding them."""
    if not checkpoint_path:
        return {"checkpoint": None, "loaded_tensors": 0, "skipped_tensors": 0}
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")
    model_state = model.state_dict()
    compatible = {
        key: value for key, value in checkpoint.items()
        if key in model_state and hasattr(value, "shape") and value.shape == model_state[key].shape
    }
    skipped = len(checkpoint) - len(compatible)
    incompatible = model.load_state_dict(compatible, strict=False)
    result = {
        "checkpoint": str(path),
        "loaded_tensors": len(compatible),
        "skipped_tensors": skipped,
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }
    print(f"Pretrained checkpoint: {result}")
    return result


def build_episode_split(data_cfg, val_episodes_per_group: int, seed: int):
    """Split episodes by (task_index, easy/hard) without frame leakage."""
    dataset_path = Path(data_cfg.data_root_dir) / data_cfg.dataset_name
    groups = defaultdict(list)
    for parquet_path in sorted((dataset_path / "data").glob("chunk-*/*.parquet")):
        frame_data = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        for episode_index, episode_data in frame_data.groupby("episode_index"):
            episode_index = int(episode_index)
            event_path = dataset_path / "meta" / "vp_events" / f"episode_{episode_index:06d}.json"
            event = json.loads(event_path.read_text(encoding="utf-8"))
            difficulty = str(event.get("notes", "")).strip().lower()
            if difficulty not in {"easy", "hard"}:
                raise ValueError(f"Episode {episode_index}: expected notes=easy/hard, got {difficulty!r}")
            groups[(int(episode_data.iloc[0]["task_index"]), difficulty)].append(episode_index)

    rng = np.random.RandomState(seed)
    train_episodes = set()
    val_episodes = set()
    group_summary = []
    for group, episodes in sorted(groups.items()):
        shuffled = sorted(set(episodes))
        rng.shuffle(shuffled)
        n_val = min(val_episodes_per_group, len(shuffled) - 1)
        val_episodes.update(shuffled[:n_val])
        train_episodes.update(shuffled[n_val:])
        group_summary.append({
            "task_index": group[0],
            "difficulty": group[1],
            "total": len(shuffled),
            "train": len(shuffled) - n_val,
            "val": n_val,
        })
    if not train_episodes or not val_episodes:
        raise ValueError("Train/validation split produced an empty partition")
    return train_episodes, val_episodes, group_summary


def filter_single_dataset(single_dataset, allowed_episodes: set[int]) -> None:
    """Restrict a loaded LeRobot dataset to an episode whitelist."""
    mask = np.asarray([int(item) in allowed_episodes for item in single_dataset._trajectory_ids])
    single_dataset._trajectory_ids = single_dataset._trajectory_ids[mask]
    single_dataset._trajectory_lengths = single_dataset._trajectory_lengths[mask]
    single_dataset._all_steps = [
        (trajectory_id, step)
        for trajectory_id, step in single_dataset._all_steps
        if int(trajectory_id) in allowed_episodes
    ]
    single_dataset.curr_traj_data = None
    single_dataset.curr_traj_id = None


def build_dataset(cfg, mode: str, allowed_episodes: set[int]):
    data_cfg = cfg.datasets.vla_data
    single_dataset = make_VisualPromptSingleDataset(
        Path(data_cfg.data_root_dir),
        data_cfg.dataset_name,
        data_cfg.robot_type,
        delete_pause_frame=bool(data_cfg.get("delete_pause_frame", False)),
        data_cfg=data_cfg,
    )
    filter_single_dataset(single_dataset, allowed_episodes)
    mixture = VisualPromptMixtureDataset(
        [(single_dataset, 1.0)],
        mode=mode,
        seed=int(cfg.seed),
        data_cfg=data_cfg,
        visual_prompt_dir=data_cfg.visual_prompt_dir,
        target_object_prompt_type=data_cfg.target_object_prompt_type,
        target_location_prompt_type=data_cfg.target_location_prompt_type,
        feed_both_images=bool(data_cfg.get("feed_both_images", False)),
    )
    return single_dataset, mixture


def save_checkpoint(
    accelerator: Accelerator, model, output_dir: Path, step: int, cfg, name: str | None = None
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = name or f"steps_{step:06d}_pytorch_model.pt"
        torch.save(
            merged_state_dict(accelerator.unwrap_model(model)),
            checkpoint_dir / filename,
        )
        OmegaConf.save(cfg, output_dir / "config.yaml")
    accelerator.wait_for_everyone()


def save_run_metadata(dataset, output_dir: Path, cfg) -> None:
    """Save config and normalization statistics required for deployment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")


def compute_grounding_loss(model, batch, cfg, device):
    grounding_samples = [sample["vp_grounding"] for sample in batch if "vp_grounding" in sample]
    if not grounding_samples:
        return torch.tensor(0.0, device=device)
    vp_images = [[sample["image"]] for sample in grounding_samples]
    vp_text = [
        f"{sample['instruction']}\nAnswer: {sample['answer']}"
        for sample in grounding_samples
    ]
    vp_inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=vp_images, instructions=vp_text)
    if "labels" not in vp_inputs:
        vp_inputs["labels"] = vp_inputs["input_ids"].clone()
        pad_id = model.qwen_vl_interface.processor.tokenizer.pad_token_id
        if pad_id is not None:
            vp_inputs["labels"][vp_inputs["labels"] == pad_id] = -100
    vp_output = model.qwen_vl_interface(**vp_inputs)
    return vp_output.loss * float(cfg.trainer.loss_scale.visual_prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--validate_model", action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--pretrained_checkpoint", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    if args.max_train_steps is not None:
        cfg.trainer.max_train_steps = args.max_train_steps
    if args.pretrained_checkpoint is not None:
        cfg.trainer.pretrained_checkpoint = args.pretrained_checkpoint

    output_dir = Path(cfg.run_root_dir) / cfg.run_id
    cfg.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_episodes, val_episodes, group_summary = build_episode_split(
        cfg.datasets.vla_data,
        int(cfg.trainer.val_episodes_per_group),
        int(cfg.trainer.val_seed),
    )
    train_single_dataset, train_dataset = build_dataset(cfg, "train", train_episodes)
    val_single_dataset, val_dataset = build_dataset(cfg, "val", val_episodes)
    save_run_metadata(train_dataset, output_dir, cfg)
    split_manifest = {
        "seed": int(cfg.trainer.val_seed),
        "val_episodes_per_group": int(cfg.trainer.val_episodes_per_group),
        "train_episodes": sorted(train_episodes),
        "val_episodes": sorted(val_episodes),
        "groups": group_summary,
    }
    (output_dir / "episode_split.json").write_text(
        json.dumps(split_manifest, indent=4) + "\n", encoding="utf-8"
    )
    print(f"SO101 train dataset: {train_single_dataset}")
    print(f"SO101 val dataset: {val_single_dataset}")
    print(f"Visual prompt train/val lengths: {len(train_dataset)}/{len(val_dataset)}")
    print(f"Visual prompt root: {cfg.datasets.vla_data.visual_prompt_dir}")
    sample = train_dataset[0]
    sample_summary = {
        "action_shape": tuple(sample["action"].shape),
        "state_shape": tuple(sample["state"].shape) if "state" in sample else None,
        "num_images": len(sample["image"]),
        "has_vp_grounding": "vp_grounding" in sample,
        "lang": sample["lang"],
    }
    print("Sample:", sample_summary)
    expected_action_dim = int(cfg.framework.action_model.action_dim)
    if sample_summary["action_shape"][-1] != expected_action_dim:
        raise ValueError(f"Action dim mismatch: {sample_summary['action_shape']} vs {expected_action_dim}")
    include_state = bool(cfg.datasets.vla_data.get("include_state", False))
    if include_state:
        expected_state_dim = int(cfg.framework.action_model.state_dim)
        if sample_summary["state_shape"][-1] != expected_state_dim:
            raise ValueError(f"State dim mismatch: {sample_summary['state_shape']} vs {expected_state_dim}")
    expected_images = 3 if bool(cfg.datasets.vla_data.get("feed_both_images", False)) else 2
    if sample_summary["num_images"] != expected_images:
        raise ValueError(f"Expected {expected_images} model images, got {sample_summary['num_images']}")
    if args.validate_only:
        return

    if args.validate_model:
        model = build_framework(cfg)
        checkpoint_summary = load_compatible_checkpoint(
            model, cfg.trainer.get("pretrained_checkpoint", None)
        )
        lora_cfg = cfg.trainer.get("vision_lora", {})
        lora_targets = []
        if bool(lora_cfg.get("enabled", False)):
            lora_targets = inject_vision_lora(
                model,
                str(lora_cfg.target_pattern),
                int(lora_cfg.rank),
                float(lora_cfg.alpha),
                float(lora_cfg.dropout),
            )
        print(
            f"Model validation passed: {checkpoint_summary}; "
            f"vision_lora_targets={len(lora_targets)}"
        )
        return

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.trainer.gradient_accumulation_steps),
        mixed_precision="bf16" if bool(cfg.trainer.enable_mixed_precision_training) else "no",
    )
    torch.manual_seed(int(cfg.seed))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(cfg.datasets.vla_data.per_device_batch_size),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=int(cfg.datasets.vla_data.get("num_workers", 0)),
        pin_memory=True,
        persistent_workers=int(cfg.datasets.vla_data.get("num_workers", 0)) > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=int(cfg.datasets.vla_data.per_device_batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(cfg.datasets.vla_data.get("num_workers", 0)),
        pin_memory=True,
        persistent_workers=int(cfg.datasets.vla_data.get("num_workers", 0)) > 0,
    )

    model = build_framework(cfg)
    checkpoint_summary = load_compatible_checkpoint(
        model, cfg.trainer.get("pretrained_checkpoint", None)
    )
    lora_cfg = cfg.trainer.get("vision_lora", {})
    lora_targets: list[str] = []
    if bool(lora_cfg.get("enabled", False)):
        lora_targets = inject_vision_lora(
            model,
            str(lora_cfg.target_pattern),
            int(lora_cfg.rank),
            float(lora_cfg.alpha),
            float(lora_cfg.dropout),
        )
    frozen = freeze_modules(model, str(cfg.trainer.get("freeze_modules", "")))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters remain after freezing")

    if bool(cfg.trainer.get("enable_gradient_checkpointing", False)):
        qwen_model = getattr(getattr(model, "qwen_vl_interface", None), "model", None)
        if qwen_model is not None and hasattr(qwen_model, "gradient_checkpointing_enable"):
            qwen_model.gradient_checkpointing_enable()
            if hasattr(qwen_model, "enable_input_require_grads"):
                qwen_model.enable_input_require_grads()

    lora_params = [
        parameter
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in (module.lora_a.weight, module.lora_b.weight)
    ]
    lora_param_ids = {id(parameter) for parameter in lora_params}
    action_params = [
        parameter for parameter in model.action_model.parameters() if parameter.requires_grad
    ]
    action_param_ids = {id(parameter) for parameter in action_params}
    other_params = [
        parameter for parameter in trainable
        if id(parameter) not in lora_param_ids and id(parameter) not in action_param_ids
    ]
    param_groups = [
        {
            "params": action_params,
            "lr": float(cfg.trainer.learning_rate.action_model),
            "name": "action_model",
        },
        {
            "params": lora_params,
            "lr": float(cfg.trainer.learning_rate.vision_lora),
            "name": "vision_lora",
        },
    ]
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": float(cfg.trainer.learning_rate.base),
            "name": "base",
        })
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(cfg.trainer.learning_rate.base),
        betas=tuple(cfg.trainer.optimizer.betas),
        eps=float(cfg.trainer.optimizer.eps),
        weight_decay=float(cfg.trainer.optimizer.weight_decay),
    )
    scheduler = get_scheduler(
        name=str(cfg.trainer.lr_scheduler_type),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.trainer.num_warmup_steps),
        num_training_steps=int(cfg.trainer.max_train_steps),
        scheduler_specific_kwargs=OmegaConf.to_container(cfg.trainer.scheduler_specific_kwargs),
    )
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )

    if accelerator.is_main_process:
        summary = {
            "frozen_modules": frozen,
            "vision_lora_targets": lora_targets,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "gradient_accumulation_steps": accelerator.gradient_accumulation_steps,
            "effective_batch_size": int(cfg.datasets.vla_data.per_device_batch_size)
            * accelerator.num_processes
            * accelerator.gradient_accumulation_steps,
            "pretrained_checkpoint": checkpoint_summary,
            "train_episodes": len(train_episodes),
            "val_episodes": len(val_episodes),
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=4) + "\n",
            encoding="utf-8",
        )
        print(summary)

    model.train()
    optimizer.zero_grad()
    completed_steps = 0
    epoch = 0
    best_val_loss = float("inf")
    val_iterator = iter(val_dataloader)

    def evaluate_validation():
        nonlocal val_iterator
        model.eval()
        action_total = 0.0
        grounding_total = 0.0
        batches = int(cfg.trainer.val_batches)
        with torch.no_grad():
            for _ in range(batches):
                try:
                    val_batch = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(val_dataloader)
                    val_batch = next(val_iterator)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=accelerator.device.type == "cuda"
                ):
                    val_action = model.forward(val_batch)["action_loss"] * float(
                        cfg.trainer.loss_scale.vla
                    )
                    val_grounding = compute_grounding_loss(
                        model, val_batch, cfg, accelerator.device
                    )
                action_total += float(val_action.detach())
                grounding_total += float(val_grounding.detach())
        model.train()
        action_mean = action_total / batches
        grounding_mean = grounding_total / batches
        return action_mean, grounding_mean, action_mean + grounding_mean

    progress = tqdm(
        total=int(cfg.trainer.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="VP-VLA SO101",
    )

    while completed_steps < int(cfg.trainer.max_train_steps):
        train_dataset.set_epoch(epoch)
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=accelerator.device.type == "cuda",
                ):
                    output = model.forward(batch)
                    action_loss = output["action_loss"] * float(cfg.trainer.loss_scale.vla)
                accelerator.backward(action_loss)

                grounding_loss = torch.tensor(0.0, device=accelerator.device)
                grounding_frequency = int(cfg.trainer.get("grounding_frequency", 1))
                if (
                    float(cfg.trainer.loss_scale.get("visual_prompt", 0.0)) > 0
                    and completed_steps % grounding_frequency == 0
                ):
                    if any("vp_grounding" in sample for sample in batch):
                        with torch.autocast(
                            "cuda",
                            dtype=torch.bfloat16,
                            enabled=accelerator.device.type == "cuda",
                        ):
                            grounding_loss = compute_grounding_loss(
                                model, batch, cfg, accelerator.device
                            )
                        accelerator.backward(grounding_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(cfg.trainer.gradient_clipping))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            completed_steps += 1
            progress.update(1)
            if accelerator.is_main_process and completed_steps % int(cfg.trainer.logging_frequency) == 0:
                print(
                    f"step={completed_steps} action_loss={action_loss.detach().float().item():.6f} "
                    f"vp_loss={grounding_loss.detach().float().item():.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}"
                )
            if completed_steps % int(cfg.trainer.save_interval) == 0:
                save_checkpoint(accelerator, model, output_dir, completed_steps, cfg)
            if completed_steps % int(cfg.trainer.val_interval) == 0:
                val_action, val_grounding, val_total = evaluate_validation()
                if accelerator.is_main_process:
                    print(
                        f"step={completed_steps} val_action_loss={val_action:.6f} "
                        f"val_vp_loss={val_grounding:.6f} val_total_loss={val_total:.6f}"
                    )
                if val_total < best_val_loss:
                    best_val_loss = val_total
                    save_checkpoint(
                        accelerator, model, output_dir, completed_steps, cfg,
                        name="best_val_pytorch_model.pt",
                    )
                    if accelerator.is_main_process:
                        (output_dir / "best_val.json").write_text(
                            json.dumps({
                                "step": completed_steps,
                                "total_loss": val_total,
                                "action_loss": val_action,
                                "visual_prompt_loss": val_grounding,
                            }, indent=4) + "\n",
                            encoding="utf-8",
                        )
            if completed_steps >= int(cfg.trainer.max_train_steps):
                break
        epoch += 1

    save_checkpoint(accelerator, model, output_dir, completed_steps, cfg)
    if accelerator.is_main_process:
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            merged_state_dict(accelerator.unwrap_model(model)),
            final_dir / "pytorch_model.pt",
        )
    progress.close()
    accelerator.end_training()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
