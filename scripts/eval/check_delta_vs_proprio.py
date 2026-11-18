#!/usr/bin/env python3
"""Cross-check the 7D EE-delta labels against the raw LeRobot proprio/commands.

The TFDS labels are built from measured joint states, but the runtime replays
them through the hardware server's joint limits and freezes. This compares:
  limits  : demo joint range vs hardware_server safe_limits / wrist_roll freeze
  lag     : teleop joint command vs the state the follower actually reached
  gripper : demo gripper motion vs the runtime's binary open/close ramp
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Mirrors src/hardware_server.py.
SAFE_LIMITS = {
    "shoulder_pan": (-73.0, 62.0),
    "shoulder_lift": (-103.0, 92.0),
    "elbow_flex": (-95.0, 96.0),
    "wrist_flex": (0.0, 94.0),
    "wrist_roll": (-165.0, -70.0),
    "gripper": (0.0, 40.0),
}
FREEZE_WRIST_ROLL = True

DEFAULT_ROOT = REPO_ROOT / "data/lerobot/so101_pick_place_fruit"
DEFAULT_URDF = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot_root", default=str(DEFAULT_ROOT))
    p.add_argument("--files", default="0 1 2 3 4 5")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--fk_episodes", type=int, default=6, help="Episodes used for the EE-space checks.")
    return p.parse_args()


def load_frames(root: Path, file_indices: set[int]) -> list[dict]:
    episodes = []
    for path in sorted((root / "data").rglob("*.parquet")):
        index = int(path.stem.split("-")[-1])
        if index not in file_indices:
            continue
        df = pd.read_parquet(path).sort_values("index")
        for episode_index, ep in df.groupby("episode_index", sort=True):
            ep = ep.sort_values("frame_index")
            episodes.append(
                {
                    "episode": int(episode_index),
                    "state": np.stack(ep["observation.state"].values).astype(np.float64),
                    "command": np.stack(ep["action"].values).astype(np.float64),
                }
            )
    return episodes


def check_limits(episodes: list[dict]) -> None:
    print("\n=== limits: demo joint range vs hardware_server safe_limits ===")
    states = np.concatenate([ep["state"] for ep in episodes])
    for i, name in enumerate(JOINTS):
        lo, hi = SAFE_LIMITS[name]
        col = states[:, i]
        below = float(np.mean(col < lo))
        above = float(np.mean(col > hi))
        flag = "  CLIPPED" if below + above > 0.01 else ""
        print(
            f"  {name:<14} demo=[{col.min():8.2f},{col.max():8.2f}]  "
            f"limit=[{lo:8.2f},{hi:8.2f}]  out_lo={below:6.3f} out_hi={above:6.3f}{flag}"
        )
    if FREEZE_WRIST_ROLL:
        col = states[:, 4]
        print(
            f"\n  hardware_server FREEZE_WRIST_ROLL=True -> wrist_roll is pinned to its boot value.\n"
            f"  demos move wrist_roll over {col.max() - col.min():.1f} deg "
            f"(std {col.std():.1f} deg), so that motion cannot be executed."
        )


def check_lag(episodes: list[dict]) -> None:
    """The label says 'go where the arm ended up'; teleop commanded further ahead."""
    print("\n=== lag: teleop command vs the state the follower reached ===")
    lag, step, resid = [], [], []
    for ep in episodes:
        s, c = ep["state"], ep["command"]
        lag.append(np.abs(c[:-1] - s[1:]))
        step.append(np.abs(s[1:] - s[:-1]))
        resid.append(np.abs(c[:-1] - s[:-1]))
    lag = np.concatenate(lag)
    step = np.concatenate(step)
    resid = np.concatenate(resid)
    print(f"  {'joint':<14} {'|cmd_t - s_t|':>14} {'|s_t+1 - s_t|':>14} {'|cmd_t - s_t+1|':>16} {'tracked':>9}")
    for i, name in enumerate(JOINTS):
        tracked = step[:, i].mean() / resid[:, i].mean() if resid[:, i].mean() > 0 else float("nan")
        print(
            f"  {name:<14} {resid[:, i].mean():14.3f} {step[:, i].mean():14.3f} "
            f"{lag[:, i].mean():16.3f} {tracked:9.3f}"
        )
    print("  tracked = fraction of the commanded gap the follower closes in one 10 Hz tick.")


def check_gripper(episodes: list[dict], open_threshold_deg: float = 20.0) -> None:
    print("\n=== gripper: demo motion vs runtime binary ramp ===")
    closing, opening, open_pos = [], [], []
    transitions = 0
    for ep in episodes:
        g_state = ep["state"][:, 5]
        g_cmd = ep["command"][:, 5]
        d = np.diff(g_state)
        closing.append(-d[d < -0.5])
        opening.append(d[d > 0.5])
        open_pos.append(g_state[g_cmd >= open_threshold_deg])
        binary = (g_cmd >= open_threshold_deg).astype(int)
        transitions += int(np.sum(np.abs(np.diff(binary))))
    closing = np.concatenate(closing)
    opening = np.concatenate(opening)
    open_pos = np.concatenate(open_pos)
    print(f"  demo close speed  mean={closing.mean():6.2f} p95={np.quantile(closing,.95):6.2f} deg/tick")
    print(f"  demo open  speed  mean={opening.mean():6.2f} p95={np.quantile(opening,.95):6.2f} deg/tick")
    print(f"  runtime close ramp = 2.00 deg/tick, open jumps straight to 40.00 deg")
    print(f"  demo gripper position while label=open: mean={open_pos.mean():6.2f} max={open_pos.max():6.2f} deg")
    print(f"  open/close transitions across {len(episodes)} episodes: {transitions}")


def check_ee_effect(episodes: list[dict], urdf: str, n_episodes: int) -> None:
    """What does freezing wrist_roll / clipping cost in EE space?"""
    from replay_gt_action_pipeline import GenesisFK, rot_angle_deg

    fk = GenesisFK(urdf)
    print("\n=== ee: cost of the hardware-server overrides in EE space ===")

    clip_pos, clip_rot, freeze_pos, freeze_rot, label_mm = [], [], [], [], []
    for ep in episodes[:n_episodes]:
        s = ep["state"]
        lock = s[0, 4]
        for t in range(len(s) - 1):
            q = s[t + 1]
            pos_ref, rot_ref = fk.pose(q)

            q_clip = q.copy()
            for i, name in enumerate(JOINTS):
                lo, hi = SAFE_LIMITS[name]
                q_clip[i] = np.clip(q_clip[i], lo, hi)
            pos_c, rot_c = fk.pose(q_clip)
            clip_pos.append(np.linalg.norm(pos_c - pos_ref) * 1000.0)
            clip_rot.append(rot_angle_deg(rot_c, rot_ref))

            q_freeze = q.copy()
            q_freeze[4] = lock
            pos_f, rot_f = fk.pose(q_freeze)
            freeze_pos.append(np.linalg.norm(pos_f - pos_ref) * 1000.0)
            freeze_rot.append(rot_angle_deg(rot_f, rot_ref))

            pos_prev, _ = fk.pose(s[t])
            label_mm.append(np.linalg.norm(pos_ref - pos_prev) * 1000.0)

    def show(name, arr, unit):
        a = np.array(arr)
        print(f"  {name:<28} mean={a.mean():8.3f} p95={np.quantile(a,.95):8.3f} max={a.max():8.3f} {unit}")

    show("label step size", label_mm, "mm")
    show("safe_limits clipping (pos)", clip_pos, "mm")
    show("safe_limits clipping (rot)", clip_rot, "deg")
    show("wrist_roll freeze (pos)", freeze_pos, "mm")
    show("wrist_roll freeze (rot)", freeze_rot, "deg")


def main() -> None:
    args = parse_args()
    files = {int(tok) for tok in args.files.replace(",", " ").split()}
    episodes = load_frames(Path(args.lerobot_root), files)
    total = sum(len(ep["state"]) for ep in episodes)
    print(f"Loaded {len(episodes)} episodes / {total} frames from files {sorted(files)}")

    check_limits(episodes)
    check_lag(episodes)
    check_gripper(episodes)
    check_ee_effect(episodes, args.urdf, args.fk_episodes)


if __name__ == "__main__":
    main()
