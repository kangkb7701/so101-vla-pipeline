#!/usr/bin/env python3
"""Build a LeRobot ACT dataset with a 3-way basket goal input.

The derived dataset keeps source shards file-000 through file-005 and adds
``observation.environment_state`` to every frame:

    blue   (task_index=0) -> [1, 0, 0]
    yellow (task_index=1) -> [0, 1, 0]
    green  (task_index=2) -> [0, 0, 1]

LeRobot ACT treats this feature as a separate environment-state token, so the
installed ACT model does not need to be patched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


GOAL_FEATURE = "observation.environment_state"
GOAL_NAMES = ["basket_blue", "basket_yellow", "basket_green"]
TASK_TO_GOAL = {
    0: np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    1: np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    2: np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/lerobot/so101_pick_place_fruit")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lerobot/so101_pick_place_fruit_act/file000_005_goal_onehot"),
    )
    parser.add_argument("--first_file", type=int, default=0)
    parser.add_argument("--last_file", type=int, default=5)
    parser.add_argument(
        "--copy_videos", action="store_true", help="Copy videos instead of using hard links."
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path, copy_file: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_file:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def add_goal_column(data: pd.DataFrame) -> pd.DataFrame:
    unknown_tasks = sorted(set(data["task_index"].astype(int)) - set(TASK_TO_GOAL))
    if unknown_tasks:
        raise ValueError(f"No one-hot mapping for task indices: {unknown_tasks}")

    result = data.copy()
    result[GOAL_FEATURE] = [TASK_TO_GOAL[int(task)].copy() for task in result["task_index"]]
    return result


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    file_indices = list(range(args.first_file, args.last_file + 1))

    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output}. Use --force to replace it.")
        shutil.rmtree(output)

    source_info = json.loads((source / "meta/info.json").read_text())
    episode_meta_frames = []
    task_episode_counts: dict[int, int] = {}
    total_frames = 0

    for file_index in file_indices:
        filename = f"file-{file_index:03d}.parquet"
        source_data_path = source / "data/chunk-000" / filename
        source_episode_path = source / "meta/episodes/chunk-000" / filename
        if not source_data_path.exists() or not source_episode_path.exists():
            raise FileNotFoundError(f"Missing source shard {filename}")

        data = add_goal_column(pd.read_parquet(source_data_path))
        destination_data_path = output / "data/chunk-000" / filename
        destination_data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(destination_data_path, index=False)
        total_frames += len(data)

        per_episode_tasks = data.groupby("episode_index")["task_index"].first()
        for task_index, count in per_episode_tasks.value_counts().items():
            task = int(task_index)
            task_episode_counts[task] = task_episode_counts.get(task, 0) + int(count)

        episode_meta_frames.append(pd.read_parquet(source_episode_path))

        for video_key, feature in source_info["features"].items():
            if feature["dtype"] != "video":
                continue
            video_name = f"file-{file_index:03d}.mp4"
            link_or_copy(
                source / "videos" / video_key / "chunk-000" / video_name,
                output / "videos" / video_key / "chunk-000" / video_name,
                args.copy_videos,
            )

    episode_meta = pd.concat(episode_meta_frames, ignore_index=True)
    episodes_path = output / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episode_meta.to_parquet(episodes_path, index=False)

    shutil.copy2(source / "meta/tasks.parquet", output / "meta/tasks.parquet")
    shutil.copy2(source / "meta/stats.json", output / "meta/stats.json")

    total_episodes = int(episode_meta["episode_index"].nunique())
    info = dict(source_info)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(task_episode_counts)
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["features"] = dict(source_info["features"])
    info["features"][GOAL_FEATURE] = {
        "dtype": "float32",
        "shape": [3],
        "names": GOAL_NAMES,
    }
    (output / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import recompute_stats

    dataset = LeRobotDataset("local/so101_pick_place_fruit_file000_005_goal_onehot", root=output)
    recompute_stats(dataset, skip_image_video=True)

    print(f"Built: {output}")
    print(f"Files: {file_indices}")
    print(f"Episodes: {total_episodes}, frames: {total_frames}")
    print(f"Episodes per task: {dict(sorted(task_episode_counts.items()))}")
    print(f"Goal feature: {GOAL_FEATURE} {GOAL_NAMES}")


if __name__ == "__main__":
    main()
