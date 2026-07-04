#!/usr/bin/env python3
"""Run main_act.py with a blue/yellow/green basket goal one-hot input."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from so101_pipeline.runtime import main_act


GOAL_FEATURE = "observation.environment_state"
GOAL_ONE_HOT = {
    "blue": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    "yellow": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    "green": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--goal", choices=tuple(GOAL_ONE_HOT), required=True)
    goal_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    original_build_dataset_frame = main_act.build_dataset_frame
    def build_dataset_frame_with_goal(ds_features, values, prefix):
        if GOAL_FEATURE in ds_features:
            values = dict(values)
            goal_names = ds_features[GOAL_FEATURE].get("names", [])
            values.update(dict(zip(goal_names, GOAL_ONE_HOT[goal_args.goal], strict=True)))
        return original_build_dataset_frame(ds_features, values, prefix)

    main_act.build_dataset_frame = build_dataset_frame_with_goal
    print(f"ACT goal: {goal_args.goal} one_hot={GOAL_ONE_HOT[goal_args.goal].tolist()}")
    main_act.main()


if __name__ == "__main__":
    main()
