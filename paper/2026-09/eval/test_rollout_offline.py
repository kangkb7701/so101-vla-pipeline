"""Offline end-to-end test of eval_rollout.py through the real `lerobot-rollout` code path (no hardware).

A fake SO-101 replays one recorded episode: get_observation() returns the recorded top/side frames (as camera1 /
camera2 by default) and joint state of the next frame, send_action() stores what rollout dispatched.
The test checks that 6 joint targets reach the robot and compares them with the recorded joint actions.

Usage (from 2026-09-paper/eval): python test_rollout_offline.py <checkpoint> [episode] [seconds] [cam_top] [cam_side]
"""

import json
import os
import sys
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

import eval_rollout  # noqa: F401  (patches build_rollout_context with the IK hook)
import lerobot.rollout.context as rollout_context
import lerobot.scripts.lerobot_rollout as lerobot_rollout
from lerobot.datasets import LeRobotDataset
from lerobot.robots import Robot, RobotConfig

ROOT = "/home/aivlab/kkb_capstone/data/lerobot/so101_pick_place_fruit_180"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SENT: list[dict] = []
GT: list[np.ndarray] = []
POLICY_PHASE = {"on": True}  # False once teardown starts (rollout then drives the arm back to its start pose)


@RobotConfig.register_subclass("fake_so101_replay")
@dataclass
class FakeSO101Config(RobotConfig):
    episode: int = 0
    cam_top: str = "camera1"
    cam_side: str = "camera2"
    cameras: dict = field(default_factory=dict)


class FakeSO101(Robot):
    config_class = FakeSO101Config
    name = "so_follower"

    def __init__(self, config: FakeSO101Config):
        super().__init__(config)
        self.config = config
        self.ds = LeRobotDataset("local/so101_pick_place_fruit_180", root=ROOT, episodes=[config.episode])
        self.t = 0
        self.cameras = {}
        self._connected = False

    @cached_property
    def observation_features(self) -> dict:
        return {**{f"{j}.pos": float for j in JOINTS},
                self.config.cam_top: (480, 640, 3), self.config.cam_side: (480, 640, 3)}

    @cached_property
    def action_features(self) -> dict:
        return {f"{j}.pos": float for j in JOINTS}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self):
        item = self.ds[min(self.t, len(self.ds) - 1)]
        self._last = item
        obs = {f"{j}.pos": float(v) for j, v in zip(JOINTS, item["observation.state"].tolist())}
        for key, cam in (("observation.images.top", self.config.cam_top), ("observation.images.wrist", self.config.cam_side)):
            obs[cam] = (item[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return obs

    def send_action(self, action):
        if POLICY_PHASE["on"]:
            SENT.append(dict(action))
            GT.append(self._last["action"].numpy())
            self.t += 1
        return action

    def disconnect(self) -> None:
        self._connected = False


def make_robot(cfg):
    return FakeSO101(cfg) if isinstance(cfg, FakeSO101Config) else _orig_make_robot(cfg)


_orig_make_robot = rollout_context.make_robot_from_config
rollout_context.make_robot_from_config = make_robot
_orig_create_strategy = lerobot_rollout.create_strategy


def create_strategy(cfg):
    strategy = _orig_create_strategy(cfg)
    teardown = strategy.teardown

    def teardown_and_mark(ctx):
        POLICY_PHASE["on"] = False
        return teardown(ctx)

    strategy.teardown = teardown_and_mark
    return strategy


lerobot_rollout.create_strategy = create_strategy

if __name__ == "__main__":
    ckpt = sys.argv[1]
    episode = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seconds = sys.argv[3] if len(sys.argv) > 3 else "6"
    cam_top = sys.argv[4] if len(sys.argv) > 4 else "camera1"
    cam_side = sys.argv[5] if len(sys.argv) > 5 else "camera2"
    cfg = json.load(open(f"{ckpt}/config.json"))
    ds = LeRobotDataset("local/so101_pick_place_fruit_180", root=ROOT, episodes=[episode])
    task = ds.meta.tasks.index[int(ds[0]["task_index"])]
    sys.argv = [
        "eval_rollout", "--strategy.type=base", f"--policy.path={ckpt}",
        "--robot.type=fake_so101_replay", f"--robot.episode={episode}",
        f"--robot.cam_top={cam_top}", f"--robot.cam_side={cam_side}",
        f"--task={task}", f"--duration={seconds}", "--fps=10",
    ] + ([f"--device={os.environ['ROLLOUT_DEVICE']}"] if os.environ.get("ROLLOUT_DEVICE") else [])
    lerobot_rollout.main()
    sent = np.array([[a[f"{j}.pos"] for j in JOINTS] for a in SENT])
    gt = np.array(GT)
    print(f"RESULT ckpt={ckpt} action_dim_policy={cfg['output_features']['action']['shape']} steps_sent={len(sent)}")
    print("keys sent:", sorted(SENT[0].keys()) if SENT else None)
    if len(sent):
        print("MAE vs recorded action (deg):", dict(zip(JOINTS, np.round(np.abs(sent - gt).mean(0), 2).tolist())))
