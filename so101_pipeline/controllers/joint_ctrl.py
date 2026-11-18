import numpy as np
from .ik_ctrl import IKController


class JointDeltaController(IKController):
    """
    Execute a 6D joint-space action: [d_shoulder_pan, d_shoulder_lift, d_elbow_flex,
    d_wrist_flex, d_wrist_roll, gripper]. Arm deltas are degrees relative to the
    measured joint state (dataset builder mode rel_cmd / rel_state); the gripper is
    an absolute command in degrees (gripper_mode="absolute"). No IK.

    wrist_roll is held at its measured value: the joint dataset zeroes it and the
    hardware server freezes it.
    """

    def __init__(self, action_scaling=1.0, max_delta_deg=20.0, **gripper_kwargs):
        super().__init__(robot=None, ee_link=None, action_scaling=action_scaling, **gripper_kwargs)
        self.max_delta_deg = float(max_delta_deg)

    def get_joint_targets(self, raw_action, current_state):
        q_deg = np.rad2deg(np.asarray(current_state["q"], dtype=np.float64))
        delta = np.asarray(raw_action[:5], dtype=np.float64) * self.action_scaling
        delta = np.clip(delta, -self.max_delta_deg, self.max_delta_deg)
        delta[self.wrist_roll_index] = 0.0

        q_target_deg = q_deg.copy()
        q_target_deg[:5] += delta
        print(
            f"📐 joint_delta={np.round(delta[:4], 2).tolist()}deg | "
            f"target={np.round(q_target_deg[:4], 1).tolist()}"
        )

        gripper_target = self._gripper_target(float(raw_action[5]), current_state)
        return np.deg2rad(q_target_deg), gripper_target
