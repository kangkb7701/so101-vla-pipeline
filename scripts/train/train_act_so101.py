#!/usr/bin/env python3
"""Train or fine-tune an ACT policy on a LeRobot SO-101 dataset.

This wrapper exists so the project has a stable command surface even if the
underlying LeRobot CLI is verbose. It delegates training to lerobot-train.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def str_bool(value: bool) -> str:
    return "true" if value else "false"


def build_command(args: argparse.Namespace) -> list[str]:
    train_bin = args.lerobot_train_bin or os.environ.get(
        "LEROBOT_TRAIN_BIN",
        "/home/aivlab/anaconda3/envs/lerobot/bin/lerobot-train",
    )
    cmd = [
        train_bin,
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--policy.device={args.device}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.push_to_hub={str_bool(args.push_to_hub)}",
        f"--wandb.enable={str_bool(args.wandb)}",
    ]
    if str(args.temporal_ensemble_coeff).lower() not in {"none", "null", ""}:
        cmd.append(f"--policy.temporal_ensemble_coeff={args.temporal_ensemble_coeff}")
    if args.dataset_root:
        cmd.append(f"--dataset.root={args.dataset_root}")
    if args.policy_path:
        cmd.append(f"--policy.path={args.policy_path}")
    else:
        cmd.append("--policy.type=act")
    if args.learning_rate is not None:
        cmd.append(f"--policy.optimizer_lr={args.learning_rate}")
        cmd.append(f"--policy.optimizer_lr_backbone={args.learning_rate}")
    if args.weight_decay is not None:
        cmd.append(f"--policy.optimizer_weight_decay={args.weight_decay}")
    if args.resume:
        cmd.append("--resume=true")
    if args.no_policy_preset:
        cmd.append("--use_policy_training_preset=false")
    cmd.extend(args.extra_args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_repo_id", required=True)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default="outputs/train/act_so101_pick_place")
    parser.add_argument("--job_name", default="act_so101_pick_place")
    parser.add_argument("--policy_path", default=None, help="Existing ACT checkpoint/config to fine-tune. Omit to train ACT from scratch.")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--n_action_steps", type=int, default=100)
    parser.add_argument("--temporal_ensemble_coeff", default="None", help="Use None for off, or e.g. 0.01 with n_action_steps=1.")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_policy_preset", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_path", default=None, help="Optional dataset summary path from prepare_act_dataset.py")
    parser.add_argument("--lerobot_train_bin", default=None)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional raw lerobot-train overrides after '--'.")
    args = parser.parse_args()

    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]

    cmd = build_command(args)
    command_text = " \\\n  ".join(shlex.quote(part) for part in cmd)
    command_dir = Path("outputs/act_train_commands")
    command_dir.mkdir(parents=True, exist_ok=True)
    command_file = command_dir / f"{args.job_name}.sh"
    command_file.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command_text + "\n", encoding="utf-8")
    command_file.chmod(0o755)

    print(command_text)
    print(f"saved command: {command_file}")
    if args.summary_path:
        print(f"dataset summary: {args.summary_path}")
    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
