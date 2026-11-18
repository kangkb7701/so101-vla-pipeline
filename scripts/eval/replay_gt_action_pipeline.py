#!/usr/bin/env python3
"""Replay TFDS ground-truth actions through the runtime IK pipeline.

Model-free check of the deployment chain. The dataset already stores the
7D EE-delta labels produced by the builder's Pinocchio FK, so feeding those
labels back through IKController + Genesis IK must reproduce the original
demonstration. If it does not, the pipeline is wrong regardless of the policy.

Three checks:
  fk   : Genesis FK deltas vs the stored Pinocchio-built action labels.
  step : one-step conversion, IK fed with the ground-truth state each tick.
  free : closed-loop replay from the first state only (error accumulation).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_URDF = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
DEFAULT_TFDS_DIR = "/home/aivlab/tensorflow_datasets"
DEFAULT_DATASET = "so101_fruit_raw_3basket_h4_binary"
ARM_JOINTS = 5

# Mirrors src/hardware_server.py.
SAFE_LIMITS = [
    (-73.0, 62.0), (-103.0, 92.0), (-95.0, 96.0), (0.0, 94.0), (-165.0, -70.0), (0.0, 40.0)
]
WRIST_ROLL_INDEX = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tfds_dir", default=DEFAULT_TFDS_DIR)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--split", default="val")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--action_scaling", type=float, default=1.0)
    p.add_argument("--smoothing_alpha", type=float, default=1.0)
    p.add_argument("--min_z", type=float, default=0.06)
    p.add_argument("--gripper_mode", default="binary", choices=["binary", "delta"])
    p.add_argument("--close_motion_scale", type=float, default=1.0)
    p.add_argument(
        "--hw_emulate",
        action="store_true",
        help="Apply hardware_server safe_limits + wrist_roll freeze to the IK solution.",
    )
    p.add_argument("--save_npz", default="outputs/gt_replay/latest.npz")
    return p.parse_args()


def load_episodes(args) -> list[dict]:
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder_from_directory(
        str(Path(args.tfds_dir) / args.dataset / args.version)
    )
    skip = tfds.decode.SkipDecoding()
    ds = builder.as_dataset(
        split=f"{args.split}[:{args.episodes}]",
        decoders={"steps": {"observation": {"image_primary": skip, "image_wrist": skip}}},
    )

    episodes = []
    for ep in tfds.as_numpy(ds):
        steps = list(ep["steps"])
        episodes.append(
            {
                "state": np.stack([s["observation"]["state"] for s in steps]).astype(np.float64),
                "action": np.stack([s["action"] for s in steps]).astype(np.float64),
            }
        )
    return episodes


class GenesisFK:
    """Runtime-side kinematics: the exact shadow robot main_real2 uses."""

    def __init__(self, urdf_path: str):
        import genesis as gs
        from scipy.spatial.transform import Rotation as R

        self.R = R
        gs.init(seed=0, precision="32", logging_level="warning")
        self.scene = gs.Scene(show_viewer=False)
        self.robot = self.scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True))
        self.scene.build()
        self.ee_link = self.robot.get_link("gripper_link")

    def pose(self, q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_rad = np.deg2rad(np.asarray(q_deg, dtype=np.float64))
        self.robot.set_dofs_position(q_rad)
        pos = self.ee_link.get_pos().detach().cpu().numpy().astype(np.float64)
        quat_wxyz = self.ee_link.get_quat().detach().cpu().numpy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return pos, self.R.from_quat(quat_xyzw).as_matrix()

    def state(self, q_deg: np.ndarray) -> dict:
        pos, rot = self.pose(q_deg)
        return {"pos": pos, "rot_mat": rot, "q": np.deg2rad(np.asarray(q_deg, dtype=np.float64))}


def rot_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation as R

    return float(np.rad2deg(np.linalg.norm(R.from_matrix(r_a @ r_b.T).as_rotvec())))


def ik_residual(fk: GenesisFK, pos: np.ndarray, rot: np.ndarray, q_seed_deg: np.ndarray) -> float:
    """IK error when seeded with the exact solution: isolates solver quality."""
    from scipy.spatial.transform import Rotation as R

    quat_xyzw = R.from_matrix(rot).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    q_sol = fk.robot.inverse_kinematics(
        link=fk.ee_link, pos=pos, quat=quat_wxyz,
        init_qpos=np.deg2rad(q_seed_deg), respect_joint_limit=True,
    )
    q_sol = q_sol.detach().cpu().numpy() if hasattr(q_sol, "detach") else np.asarray(q_sol)
    pos_sol, _ = fk.pose(np.rad2deg(q_sol))
    return float(np.linalg.norm(pos_sol - pos))


def summarize(name: str, values: np.ndarray, unit: str) -> None:
    v = np.asarray(values, dtype=np.float64)
    print(
        f"  {name:<22} mean={v.mean():8.4f} p50={np.median(v):8.4f} "
        f"p95={np.quantile(v, 0.95):8.4f} max={v.max():8.4f} {unit}"
    )


def check_fk(fk: GenesisFK, episodes: list[dict]) -> None:
    """Do the runtime FK and the label-time FK describe the same robot?"""
    print("\n=== fk: Genesis FK delta vs stored label delta ===")
    pos_err, rot_err, gt_pos_norm = [], [], []
    for ep in episodes:
        states, actions = ep["state"], ep["action"]
        for t in range(len(states) - 1):
            pos_a, rot_a = fk.pose(states[t])
            pos_b, rot_b = fk.pose(states[t + 1])
            d_pos = pos_b - pos_a
            d_rot = rot_b @ rot_a.T
            label = actions[t]
            pos_err.append(np.linalg.norm(d_pos - label[:3]) * 1000.0)
            from scipy.spatial.transform import Rotation as R

            rot_err.append(rot_angle_deg(d_rot, R.from_rotvec(label[3:6]).as_matrix()))
            gt_pos_norm.append(np.linalg.norm(label[:3]) * 1000.0)
    summarize("label |d_pos| (scale)", np.array(gt_pos_norm), "mm/step")
    summarize("pos delta mismatch", np.array(pos_err), "mm")
    summarize("rot delta mismatch", np.array(rot_err), "deg")


def apply_hardware_overrides(q_deg: np.ndarray, lock_deg: float) -> np.ndarray:
    """What the follower actually receives: wrist_roll frozen, the rest clipped."""
    out = q_deg.copy()
    out[WRIST_ROLL_INDEX] = lock_deg
    for i, (lo, hi) in enumerate(SAFE_LIMITS):
        if i == WRIST_ROLL_INDEX:
            continue
        out[i] = np.clip(out[i], lo, hi)
    return out


def replay(fk: GenesisFK, controller, episodes: list[dict], free_run: bool, hw_emulate: bool = False) -> dict:
    mode = "free" if free_run else "step"
    tag = " (hardware overrides applied)" if hw_emulate else ""
    print(f"\n=== {mode}: GT action -> IKController -> joint target{tag} ===")
    joint_err, pos_err, rot_err, zclip, dclip = [], [], [], 0, 0
    oracle_pos_err, n_close = [], 0
    traces = []

    for ep_idx, ep in enumerate(episodes):
        states, actions = ep["state"], ep["action"]
        q_cur = states[0].copy()
        wrist_roll_lock = float(states[0][WRIST_ROLL_INDEX])
        ep_joint, ep_pos = [], []
        controller.reset_ghost()

        for t in range(len(states) - 1):
            if not free_run:
                q_cur = states[t].copy()
            state = fk.state(q_cur)
            action = actions[t].astype(np.float32)

            if state["pos"][2] < controller.min_z:
                zclip += 1
            if np.any(np.abs(action[:3] * controller.action_scaling) > 0.03):
                dclip += 1
            if controller.gripper_mode == "binary" and action[6] < controller.gripper_open_threshold:
                n_close += 1

            with contextlib.redirect_stdout(io.StringIO()):
                q_target, gripper_target = controller.get_joint_targets(action, state)
            if hasattr(q_target, "detach"):
                q_target = q_target.detach().cpu().numpy()
            q_target_deg = np.rad2deg(np.asarray(q_target, dtype=np.float64))
            if hw_emulate:
                q_target_deg = apply_hardware_overrides(q_target_deg, wrist_roll_lock)

            q_next_gt = states[t + 1]
            joint_err.append(np.abs(q_target_deg[:ARM_JOINTS] - q_next_gt[:ARM_JOINTS]))

            pos_pred, rot_pred = fk.pose(q_target_deg)
            pos_gt, rot_gt = fk.pose(q_next_gt)
            pos_err.append(np.linalg.norm(pos_pred - pos_gt) * 1000.0)
            rot_err.append(rot_angle_deg(rot_pred, rot_gt))
            oracle_pos_err.append(ik_residual(fk, pos_gt, rot_gt, q_next_gt) * 1000.0)

            q_cur = q_target_deg.copy()
            q_cur[5] = gripper_target
            ep_joint.append(q_target_deg.copy())
            ep_pos.append(pos_pred.copy())

        traces.append({"joints": np.array(ep_joint), "pos": np.array(ep_pos), "gt": states})
        print(
            f"  episode {ep_idx}: steps={len(ep_joint)} "
            f"final_pos_err={pos_err[-1]:.1f}mm "
            f"final_joint_err={np.max(joint_err[-1]):.2f}deg"
        )

    joint_err = np.array(joint_err)
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    for i, name in enumerate(names):
        summarize(f"joint {name}", joint_err[:, i], "deg")
    summarize("EE position error", np.array(pos_err), "mm")
    summarize("EE rotation error", np.array(rot_err), "deg")
    summarize("IK residual (oracle init)", np.array(oracle_pos_err), "mm")
    print(
        f"  min_z clip hits={zclip}  delta clip hits={dclip}  "
        f"close-intent steps={n_close}  (of {len(joint_err)} steps)"
    )
    return {"joint_err": joint_err, "pos_err": np.array(pos_err), "rot_err": np.array(rot_err), "traces": traces}


def main() -> None:
    args = parse_args()
    episodes = load_episodes(args)
    print(f"Loaded {len(episodes)} episodes from {args.dataset}[{args.split}]")
    for i, ep in enumerate(episodes):
        print(f"  episode {i}: {len(ep['state'])} steps")

    fk = GenesisFK(args.urdf)
    check_fk(fk, episodes)

    from controllers.ik_ctrl import IKController

    controller = IKController(
        robot=fk.robot,
        ee_link=fk.ee_link,
        action_scaling=args.action_scaling,
        smoothing_alpha=args.smoothing_alpha,
        min_z=args.min_z,
        gripper_mode=args.gripper_mode,
        gripper_close_motion_scale=args.close_motion_scale,
    )

    out_step = replay(fk, controller, episodes, free_run=False, hw_emulate=args.hw_emulate)
    out_free = replay(fk, controller, episodes, free_run=True, hw_emulate=args.hw_emulate)

    if args.save_npz:
        path = Path(args.save_npz)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            step_joint_err=out_step["joint_err"],
            step_pos_err=out_step["pos_err"],
            step_rot_err=out_step["rot_err"],
            free_joint_err=out_free["joint_err"],
            free_pos_err=out_free["pos_err"],
            free_rot_err=out_free["rot_err"],
        )
        print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
