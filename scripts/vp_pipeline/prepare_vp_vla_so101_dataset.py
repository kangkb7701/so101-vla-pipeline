#!/usr/bin/env python3
"""Prepare a contiguous SO-101 LeRobot v3 subset for VP-VLA training.

The source dataset is never modified. Selected parquet files are rewritten with
7D delta-EE actions matching main_real2, videos are copied or symlinked, and
VP-VLA's modality metadata is generated.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SO101_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
]
ACTION_NAMES = [
    "delta_eef.x", "delta_eef.y", "delta_eef.z",
    "delta_eef.rx", "delta_eef.ry", "delta_eef.rz", "gripper_open",
]
DEFAULT_URDF_PATH = Path("/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")


class FKConverter:
    """Convert SO-101 joint states in degrees to world-frame EE deltas."""

    def __init__(self, urdf_path: Path):
        import pinocchio as pin
        from scipy.spatial.transform import Rotation

        self.pin = pin
        self.rotation = Rotation
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.q_indices: list[int] = []
        for name in SO101_JOINT_NAMES:
            if not self.model.existJointName(name):
                raise ValueError(f"URDF does not contain joint {name!r}")
            self.q_indices.append(self.model.joints[self.model.getJointId(name)].idx_q)
        self.ee_frame_id = self.model.getFrameId("gripper_link")
        if self.ee_frame_id == len(self.model.frames):
            raise ValueError("URDF does not contain frame 'gripper_link'")

    def _ee_pose(self, joints_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.zeros(self.model.nq, dtype=np.float64)
        for q_index, value in zip(self.q_indices, np.deg2rad(joints_deg), strict=True):
            q[q_index] = value
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.ee_frame_id]
        return pose.translation.copy(), pose.rotation.copy()

    def compute_delta(self, current_deg: np.ndarray, next_deg: np.ndarray) -> np.ndarray:
        current_position, current_rotation = self._ee_pose(current_deg)
        next_position, next_rotation = self._ee_pose(next_deg)
        delta_position = next_position - current_position
        delta_rotation = self.rotation.from_matrix(next_rotation @ current_rotation.T).as_rotvec()
        return np.concatenate([delta_position, delta_rotation]).astype(np.float32)


def parse_file_indices(value: str) -> list[int]:
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise ValueError("At least one file index is required")
    return sorted(selected)


def read_parquet_files(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No parquet files selected")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def copy_or_link(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def build_modality_json() -> dict[str, Any]:
    state = {
        f"{name}.pos": {
            "start": index, "end": index + 1, "absolute": True,
            "dtype": "float32", "original_key": "observation.state",
        }
        for index, name in enumerate(SO101_JOINT_NAMES)
    }
    action = {
        "delta_eef_position": {
            "start": 0, "end": 3, "absolute": False,
            "dtype": "float32", "original_key": "action",
        },
        "delta_eef_rotation": {
            "start": 3, "end": 6, "absolute": False, "rotation_type": "axis_angle",
            "dtype": "float32", "original_key": "action",
        },
        "gripper_open": {
            "start": 6, "end": 7, "absolute": True,
            "dtype": "float32", "original_key": "action",
        },
    }
    return {
        "state": state,
        "action": action,
        "video": {
            "primary_image": {"original_key": "observation.images.top"},
            "wrist_image": {"original_key": "observation.images.wrist"},
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def transform_actions(
    data: pd.DataFrame, fk_converter: FKConverter, gripper_open_threshold_deg: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    output = data.copy()
    transformed = np.zeros((len(output), 7), dtype=np.float32)
    closed_count = 0
    for episode_positions in output.groupby("episode_index", sort=False).indices.values():
        positions = np.asarray(episode_positions, dtype=np.int64)
        positions = positions[np.argsort(output.iloc[positions]["frame_index"].to_numpy())]
        states = np.stack([
            np.asarray(value, dtype=np.float64)
            for value in output.iloc[positions]["observation.state"]
        ])
        commands = np.asarray([
            np.asarray(value, dtype=np.float32)[5] for value in output.iloc[positions]["action"]
        ])
        gripper_open = (commands >= gripper_open_threshold_deg).astype(np.float32)
        closed_count += int(np.sum(gripper_open == 0.0))
        for local_index, position in enumerate(positions):
            if local_index + 1 < len(positions):
                transformed[position, :6] = fk_converter.compute_delta(
                    states[local_index], states[local_index + 1]
                )
            transformed[position, 6] = gripper_open[local_index]
    output["action"] = [row for row in transformed]
    return output, {
        "closed_fraction": closed_count / len(output),
        "max_abs_translation_delta_m": float(np.max(np.abs(transformed[:, :3]))),
        "max_abs_rotation_delta_rad": float(np.max(np.abs(transformed[:, 3:6]))),
    }


def update_episode_action_stats(episodes: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Replace stale 6D action statistics with statistics from transformed 7D actions."""
    output = episodes.copy()
    for row_index, episode_row in output.iterrows():
        episode_index = int(episode_row["episode_index"])
        values = np.stack(
            data[data["episode_index"] == episode_index].sort_values("frame_index")["action"]
        ).astype(np.float64)
        if len(values) != int(episode_row["length"]):
            raise ValueError(f"Episode {episode_index}: action length mismatch")
        statistics = {
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "count": [len(values)],
            "q01": np.quantile(values, 0.01, axis=0),
            "q10": np.quantile(values, 0.10, axis=0),
            "q50": np.quantile(values, 0.50, axis=0),
            "q90": np.quantile(values, 0.90, axis=0),
            "q99": np.quantile(values, 0.99, axis=0),
        }
        for name, value in statistics.items():
            output.at[row_index, f"stats/action/{name}"] = value
    return output


def validate_source(
    source_root: Path, file_indices: list[int]
) -> tuple[list[Path], list[Path], pd.DataFrame, pd.DataFrame]:
    data_paths = [
        source_root / "data" / "chunk-000" / f"file-{index:03d}.parquet"
        for index in file_indices
    ]
    episode_paths = [
        source_root / "meta" / "episodes" / "chunk-000" / f"file-{index:03d}.parquet"
        for index in file_indices
    ]
    missing = [path for path in [*data_paths, *episode_paths] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing selected files: {missing}")
    data = read_parquet_files(data_paths)
    episodes = read_parquet_files(episode_paths).sort_values("episode_index").reset_index(drop=True)
    selected_episodes = sorted(int(x) for x in episodes["episode_index"].tolist())
    if selected_episodes != list(range(selected_episodes[0], selected_episodes[-1] + 1)):
        raise ValueError("Selected file indices must contain a contiguous episode range")
    if selected_episodes[0] != 0:
        raise ValueError("The subset must start at episode 0 so indices remain unchanged")
    if set(int(x) for x in data["episode_index"].unique()) != set(selected_episodes):
        raise ValueError("Data and episode metadata contain different episode sets")
    return data_paths, episode_paths, data, episodes


def validate_tasks(source_root: Path, data: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    tasks = pd.read_parquet(source_root / "meta" / "tasks.parquet")
    if tasks.index.name != "task" or "task_index" not in tasks.columns:
        raise ValueError("tasks.parquet must use task as its index and contain task_index")
    used_indices = set(int(x) for x in data["task_index"].unique())
    known_indices = set(int(x) for x in tasks["task_index"].tolist())
    if not used_indices.issubset(known_indices):
        raise ValueError(f"Unknown task indices: {sorted(used_indices - known_indices)}")
    episode_tasks = {str(task) for values in episodes["tasks"] for task in values}
    if not episode_tasks.issubset(set(str(x) for x in tasks.index.tolist())):
        raise ValueError("Episode metadata contains unknown task strings")
    return tasks[tasks["task_index"].isin(sorted(used_indices))].copy()


def write_subset(
    source_root: Path, output_root: Path, file_indices: list[int], data_paths: list[Path],
    episode_paths: list[Path], data: pd.DataFrame, episodes: pd.DataFrame, tasks: pd.DataFrame,
    fk_converter: FKConverter, gripper_open_threshold_deg: float,
    copy_videos: bool, overwrite: bool,
) -> None:
    if source_root.resolve() == output_root.resolve():
        raise ValueError("output_root must differ from source_root")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)

    action_summaries: list[tuple[int, dict[str, float]]] = []
    for src in data_paths:
        frame_data = pd.read_parquet(src)
        transformed, action_summary = transform_actions(
            frame_data, fk_converter, gripper_open_threshold_deg
        )
        dst = output_root / src.relative_to(source_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        transformed.to_parquet(dst, index=False)
        episode_src = next(path for path in episode_paths if path.name == src.name)
        episode_metadata = update_episode_action_stats(pd.read_parquet(episode_src), transformed)
        episode_dst = output_root / episode_src.relative_to(source_root)
        episode_dst.parent.mkdir(parents=True, exist_ok=True)
        episode_metadata.to_parquet(episode_dst, index=False)
        action_summaries.append((len(frame_data), action_summary))

    info = json.loads((source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    info.update(
        total_episodes=int(len(episodes)), total_frames=int(len(data)),
        total_tasks=int(len(tasks)), splits={"train": f"0:{len(episodes)}"},
    )
    info["features"]["action"] = {"dtype": "float32", "names": ACTION_NAMES, "shape": [7]}
    meta_root = output_root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
    tasks.to_parquet(meta_root / "tasks.parquet")
    (meta_root / "modality.json").write_text(
        json.dumps(build_modality_json(), indent=4) + "\n", encoding="utf-8"
    )
    source_stats = source_root / "meta" / "stats.json"
    if source_stats.exists():
        shutil.copy2(source_stats, meta_root / "stats_source_full_dataset.json")
    vp_events_src = source_root / "meta" / "vp_events"
    if vp_events_src.exists():
        for episode_index in episodes["episode_index"]:
            src = vp_events_src / f"episode_{int(episode_index):06d}.json"
            if not src.exists():
                raise FileNotFoundError(f"Missing VP event sidecar: {src}")
            copy_or_link(src, output_root / src.relative_to(source_root), copy_files=True)
    for video_key in ("observation.images.top", "observation.images.wrist"):
        for file_index in file_indices:
            src = source_root / "videos" / video_key / "chunk-000" / f"file-{file_index:03d}.mp4"
            if not src.exists():
                raise FileNotFoundError(f"Missing video file: {src}")
            copy_or_link(src, output_root / src.relative_to(source_root), copy_files=copy_videos)

    frame_counts = np.asarray([count for count, _ in action_summaries])
    summary = {
        "source_root": str(source_root), "output_root": str(output_root),
        "selected_file_indices": file_indices,
        "episode_range": [int(episodes["episode_index"].min()), int(episodes["episode_index"].max())],
        "num_episodes": int(len(episodes)), "num_frames": int(len(data)),
        "action_schema": ACTION_NAMES,
        "gripper_open_threshold_deg": gripper_open_threshold_deg,
        "closed_fraction": float(np.average(
            [item["closed_fraction"] for _, item in action_summaries], weights=frame_counts
        )),
        "max_abs_translation_delta_m": max(
            item["max_abs_translation_delta_m"] for _, item in action_summaries
        ),
        "max_abs_rotation_delta_rad": max(
            item["max_abs_rotation_delta_rad"] for _, item in action_summaries
        ),
        "tasks": [
            {"task": str(task), "task_index": int(row["task_index"])}
            for task, row in tasks.iterrows()
        ],
        "videos_copied": copy_videos,
    }
    (meta_root / "vp_vla_prepare_summary.json").write_text(
        json.dumps(summary, indent=4) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--file_indices", default="0-5", help='Comma/range list, e.g. "0-5"')
    parser.add_argument("--urdf_path", default=str(DEFAULT_URDF_PATH))
    parser.add_argument("--gripper_open_threshold_deg", type=float, default=20.0)
    parser.add_argument("--copy_videos", action="store_true", help="Copy videos instead of symlinking")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    urdf_path = Path(args.urdf_path).expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(urdf_path)
    file_indices = parse_file_indices(args.file_indices)
    data_paths, episode_paths, data, episodes = validate_source(source_root, file_indices)
    tasks = validate_tasks(source_root, data, episodes)
    write_subset(
        source_root, output_root, file_indices, data_paths, episode_paths, data, episodes, tasks,
        FKConverter(urdf_path), args.gripper_open_threshold_deg,
        copy_videos=args.copy_videos, overwrite=args.overwrite,
    )
    print(f"Prepared VP-VLA SO-101 dataset: {output_root}")
    print(f"episodes={len(episodes)} frames={len(data)} files={file_indices}")


if __name__ == "__main__":
    main()
