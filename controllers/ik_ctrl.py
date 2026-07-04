import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from .base_ctrl import BaseController


class IKController(BaseController):
    def __init__(self, robot, ee_link, action_scaling=1.0, cam_pos=None, lookat=None, smoothing_alpha=1.0,
                 resync_drift_cm=5.0, resync_rot_deg=30.0, min_z=0.06,
                 freeze_wrist_roll=False, wrist_roll_index=4, fixed_wrist_roll_rad=None,
                 gripper_mode="delta", gripper_open_pos=40.0,
                 gripper_close_min_pos=5.0, gripper_close_step_deg=2.0,
                 gripper_open_threshold=0.5, gripper_close_motion_scale=0.25):
        self.robot = robot
        self.ee_link = ee_link
        self.action_scaling = action_scaling
        self.smoothing_alpha = smoothing_alpha
        self.last_q_target = None
        self.freeze_wrist_roll = freeze_wrist_roll
        self.wrist_roll_index = wrist_roll_index
        self.fixed_wrist_roll_rad = fixed_wrist_roll_rad
        self.locked_wrist_roll_rad = None
        self.gripper_mode = str(gripper_mode).lower()
        self.gripper_open_pos = float(gripper_open_pos)
        self.gripper_close_min_pos = float(gripper_close_min_pos)
        self.gripper_close_step_deg = float(gripper_close_step_deg)
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.gripper_close_motion_scale = float(gripper_close_motion_scale)
        if self.gripper_mode not in {"delta", "binary"}:
            raise ValueError(f"Unsupported gripper_mode={gripper_mode!r}; use 'delta' or 'binary'.")

        # VP-SAM3 builder stores EE position deltas in the robot/base frame.
        # Keep this as identity unless the dataset action convention is changed.
        self.R_cam_to_base = np.eye(3)
        self.min_z = min_z

        print(
            f"🔧 IK Controller: real-state delta mode | "
            f"Action Scaling: {action_scaling} | Smoothing Alpha: {smoothing_alpha} | "
            f"Z floor: {min_z}m | Gripper mode: {self.gripper_mode} | "
            f"Close motion scale: {self.gripper_close_motion_scale}"
        )

    def _ensure_wrist_roll_lock(self, current_state):
        if not self.freeze_wrist_roll:
            return None
        if self.locked_wrist_roll_rad is not None:
            return self.locked_wrist_roll_rad

        if self.fixed_wrist_roll_rad is not None:
            self.locked_wrist_roll_rad = float(self.fixed_wrist_roll_rad)
        else:
            q = current_state.get("q") if current_state is not None else None
            if q is None or len(q) <= self.wrist_roll_index:
                return None
            self.locked_wrist_roll_rad = float(q[self.wrist_roll_index])

        print(
            f"🔒 wrist_roll freeze 활성화 | "
            f"index={self.wrist_roll_index} "
            f"lock={np.rad2deg(self.locked_wrist_roll_rad):.2f}°"
        )
        return self.locked_wrist_roll_rad

    def _freeze_wrist_roll_array(self, q_array, current_state=None):
        if not self.freeze_wrist_roll or q_array is None:
            return q_array
        lock = self._ensure_wrist_roll_lock(current_state)
        if lock is None or len(q_array) <= self.wrist_roll_index:
            return q_array
        q_array[self.wrist_roll_index] = lock
        return q_array

    def reset_ghost(self):
        """Compatibility hook for old callers; this controller no longer keeps ghost targets."""
        self.last_q_target = None

    def _current_gripper_pos(self, current_state):
        gripper_tel = current_state.get("gripper_telemetry") or {}
        gripper_pos = gripper_tel.get("pos")
        if gripper_pos is None and len(current_state.get("q", [])) > 5:
            gripper_pos = float(np.rad2deg(current_state["q"][5]))
        if gripper_pos is None:
            gripper_pos = 39.4  # selected VP dataset start median action_first5
        return float(gripper_pos)

    def get_joint_targets(self, raw_action, current_state):
        """
        Execute the learned 7D action with the same one-step convention used in
        the TFDS builder: target_t = real_state_t + delta_t.

        raw_action: [dx, dy, dz, drx, dry, drz, gripper]
        current_state: {"pos": [3], "rot_mat": [3,3], "q": joint array}
        """
        self._ensure_wrist_roll_lock(current_state)

        current_pos = current_state["pos"].copy()
        current_rot = current_state["rot_mat"].copy()
        current_q = self._freeze_wrist_roll_array(current_state["q"].copy(), current_state)
        gripper_signal = float(raw_action[6])
        close_intent = (
            self.gripper_mode == "binary"
            and gripper_signal < self.gripper_open_threshold
        )
        motion_scale = self.gripper_close_motion_scale if close_intent else 1.0

        # 1. Position: dataset action is one-step EE delta, so apply it from
        # the measured real EE pose every control tick instead of accumulating
        # against an internal ghost pose.
        delta_pos = raw_action[:3].astype(np.float32) * self.action_scaling * motion_scale
        delta_pos = np.clip(delta_pos, -0.03, 0.03)
        target_pos = current_pos + (self.R_cam_to_base @ delta_pos)
        if target_pos[2] < self.min_z:
            target_pos[2] = self.min_z

        real_delta_cm = np.linalg.norm(target_pos - current_pos) * 100.0
        print(
            f"📏 target_delta={real_delta_cm:5.2f}cm | "
            f"target={np.round(target_pos, 3).tolist()} "
            f"real={np.round(current_pos, 3).tolist()}"
            + (f" | close_scale={motion_scale:.2f}" if close_intent else "")
        )

        # 2. Rotation: builder convention is R_delta = R_next @ R_curr.T,
        # so reconstruct R_next with left multiplication from the real pose.
        delta_rotvec = raw_action[3:6].astype(np.float32) * motion_scale
        delta_rot_mat = R.from_rotvec(delta_rotvec).as_matrix()
        target_rot_mat = delta_rot_mat @ current_rot
        delta_rot_deg = np.rad2deg(np.linalg.norm(delta_rotvec))
        target_rotvec_deg = np.rad2deg(R.from_matrix(target_rot_mat).as_rotvec())
        real_rotvec_deg = np.rad2deg(R.from_matrix(current_rot).as_rotvec())
        print(
            f"🔄 target_rot_delta={delta_rot_deg:5.1f}° | "
            f"target_rv={np.round(target_rotvec_deg, 1).tolist()}° "
            f"real_rv={np.round(real_rotvec_deg, 1).tolist()}°"
        )

        target_quat_xyzw = R.from_matrix(target_rot_mat).as_quat()
        target_quat_wxyz = np.array([
            target_quat_xyzw[3],
            target_quat_xyzw[0],
            target_quat_xyzw[1],
            target_quat_xyzw[2],
        ])

        _t_ik_start = time.time()
        q_target = self.robot.inverse_kinematics(
            link=self.ee_link,
            pos=target_pos,
            quat=target_quat_wxyz,
            init_qpos=current_q,
            respect_joint_limit=True,
        )
        _t_ik_solver_end = time.time()

        q_target_device = q_target.device if hasattr(q_target, "device") else None
        if hasattr(q_target, "detach"):
            q_target_np = q_target.detach().cpu().numpy()
        else:
            q_target_np = np.asarray(q_target, dtype=np.float32)
        q_target_np = self._freeze_wrist_roll_array(q_target_np, current_state)
        _t_cpu_sync_end = time.time()

        _ik_solver_ms = (_t_ik_solver_end - _t_ik_start) * 1000.0
        _cpu_sync_ms = (_t_cpu_sync_end - _t_ik_solver_end) * 1000.0
        print(f"⏱️ [IK] solver={_ik_solver_ms:5.1f}ms cpu_sync={_cpu_sync_ms:5.1f}ms")

        # Optional joint-target EMA. Default 1.0 means no smoothing, preserving
        # the learned one-step action semantics as closely as possible.
        if self.smoothing_alpha < 1.0:
            if self.last_q_target is None:
                self.last_q_target = current_q.copy()
            q_target_np = (
                self.smoothing_alpha * q_target_np
                + (1.0 - self.smoothing_alpha) * self.last_q_target
            )
            q_target_np = self._freeze_wrist_roll_array(q_target_np, current_state)
        self.last_q_target = q_target_np.copy()

        if q_target_device is not None:
            q_target = torch.from_numpy(q_target_np).to(q_target_device)
        else:
            q_target = q_target_np

        # 3. Gripper.
        # - delta: old checkpoints output a one-step gripper delta in degrees.
        # - binary: VP-style binary checkpoints output open intent (1=open, 0=close).
        #   Closing is applied gradually from the measured real position to avoid
        #   slamming the gripper to the hard limit.
        current_gripper = self._current_gripper_pos(current_state)
        if self.gripper_mode == "binary":
            if gripper_signal >= self.gripper_open_threshold:
                gripper_target = self.gripper_open_pos
                gripper_cmd = "open"
            else:
                gripper_target = max(
                    self.gripper_close_min_pos,
                    current_gripper - self.gripper_close_step_deg,
                )
                gripper_cmd = "close"
            gripper_target = float(np.clip(gripper_target, 0.0, self.gripper_open_pos))
            print(
                f"🔧 Binary Gripper: {gripper_cmd} "
                f"(signal={gripper_signal:.3f}, real={current_gripper:.2f}, "
                f"target={gripper_target:.2f})"
            )
        else:
            gripper_target = float(np.clip(current_gripper + gripper_signal, 0.0, 40.0))
            if abs(gripper_signal) > 0.01:
                print(
                    f"🔧 Real Gripper Target: {gripper_target:.4f} "
                    f"(real={current_gripper:.2f}, delta={gripper_signal:.2f} deg)"
                )

        return q_target, gripper_target
