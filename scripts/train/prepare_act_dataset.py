#!/usr/bin/env python3
"""Validate a LeRobot SO-101 dataset before ACT training.

This is intentionally a light preprocessing step: ACT trains directly from a
LeRobotDataset. The script checks that the recorded dataset has action,
proprioception, and at least one camera stream, then writes a ready-to-edit ACT
training command next to a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LEROBOT_SRC = Path("/home/aivlab/lerobot/src")
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets import LeRobotDataset  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _feature_shape(feature: dict[str, Any]) -> list[int] | None:
    shape = feature.get("shape")
    if shape is None:
        return None
    return [int(x) for x in shape]


def build_train_command(args: argparse.Namespace, summary_path: Path) -> list[str]:
    cmd = [
        "/home/aivlab/anaconda3/envs/lerobot/bin/python",
        "datasets/train_act_so101.py",
        "--dataset_repo_id",
        args.dataset_repo_id,
        "--output_dir",
        args.output_dir,
        "--job_name",
        args.job_name,
        "--steps",
        str(args.steps),
        "--batch_size",
        str(args.batch_size),
        "--chunk_size",
        str(args.chunk_size),
        "--n_action_steps",
        str(args.n_action_steps),
    ]
    if args.dataset_root:
        cmd += ["--dataset_root", args.dataset_root]
    if args.wandb:
        cmd.append("--wandb")
    cmd += ["--summary_path", str(summary_path)]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_repo_id", required=True, help="LeRobot dataset repo id, e.g. local_user/so101_pick_place_act")
    parser.add_argument("--dataset_root", default=None, help="Optional local dataset root directory")
    parser.add_argument("--out_dir", default="outputs/act_dataset_checks", help="Where to write validation summary")
    parser.add_argument("--expected_camera", action="append", default=[], help="Camera name expected in observation.images.<name>; may be repeated")
    parser.add_argument("--min_episodes", type=int, default=1)
    parser.add_argument("--job_name", default="act_so101_pick_place")
    parser.add_argument("--output_dir", default="outputs/train/act_so101_pick_place")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--n_action_steps", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        download_videos=False,
    )
    features = dataset.meta.features
    image_keys = sorted(k for k in features if k.startswith("observation.images."))
    state_key = "observation.state"
    action_key = "action"

    errors: list[str] = []
    if getattr(dataset.meta, "total_episodes", 0) < args.min_episodes:
        errors.append(f"dataset has {dataset.meta.total_episodes} episodes, expected at least {args.min_episodes}")
    if action_key not in features:
        errors.append("missing action feature")
    if state_key not in features:
        errors.append("missing observation.state feature")
    if not image_keys:
        errors.append("missing observation.images.* camera features")

    expected_image_keys = [f"observation.images.{name}" for name in args.expected_camera]
    for key in expected_image_keys:
        if key not in features:
            errors.append(f"missing expected camera feature: {key}")

    action_names = features.get(action_key, {}).get("names", [])
    state_names = features.get(state_key, {}).get("names", [])
    summary = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset.root),
        "fps": dataset.fps,
        "num_frames": len(dataset),
        "num_episodes": getattr(dataset.meta, "total_episodes", None),
        "robot_type": getattr(dataset.meta, "robot_type", None),
        "image_keys": image_keys,
        "state_shape": _feature_shape(features.get(state_key, {})),
        "state_names": state_names,
        "action_shape": _feature_shape(features.get(action_key, {})),
        "action_names": action_names,
        "features": _jsonable(features),
        "errors": errors,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.dataset_repo_id.replace("/", "__")
    summary_path = out_dir / f"{safe_name}_act_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    command_path = out_dir / f"{safe_name}_train_act_command.sh"
    cmd = build_train_command(args, summary_path)
    command_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + " \\\n  ".join(cmd) + "\n", encoding="utf-8")
    command_path.chmod(0o755)

    print(f"dataset: {args.dataset_repo_id}")
    print(f"frames={len(dataset)} episodes={summary['num_episodes']} fps={dataset.fps}")
    print(f"images={image_keys}")
    print(f"state={summary['state_shape']} action={summary['action_shape']}")
    print(f"summary: {summary_path}")
    print(f"train command: {command_path}")
    if errors:
        print("ERRORS:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
