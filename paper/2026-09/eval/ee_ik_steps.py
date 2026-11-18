"""Policy-side IK so end-effector checkpoints run through the unmodified `lerobot-rollout` loop.

EE checkpoints output [x, y, z, r1..r6, gripper] (absolute after the checkpoint's own postprocessor).
`install_ee_ik` appends an IK step to the postprocessor that turns this into the 6 joint targets the SO-101
expects, so rollout dispatch, recording and interpolation stay identical to the joint representations.

IK rules (EXPERIMENT_PLAN.md "도달 불가능 자세 처리"):
- LeRobot `RobotKinematics.inverse_kinematics`, position weight 1.0, orientation weight 0.01 (position first)
- initial guess = current measured joints (raw observation.state, cached by a preprocessor step); the placo model
  is refreshed at that guess first (LeRobot's IK does not call update_kinematics() before solving)
- clip to the URDF joint limits; non-finite solution -> keep the current joints
- per-step log of target vs achieved pose (position mm, rotation deg) to a CSV
"""

import csv
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import AddBatchDimensionProcessorStep, ProcessorStep, RelativeEEActionsProcessorStep
from lerobot.processor.relative_ee_action_processor import SO101_ARM_JOINTS, ee_vector_to_pose
from lerobot.utils.constants import OBS_STATE

DEFAULT_URDF = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
# Recorded joint-action range of so101_pick_place_fruit_180 (deg); the real arm exceeds the URDF limits
# (shoulder_lift 15%, wrist_flex 28%, wrist_roll 7.5% of frames), see EXPERIMENT_PLAN.md.
DATA_JOINT_RANGE_DEG = (
    np.array([-82.5, -106.5, -94.6, 26.5, -189.7]),
    np.array([68.9, 74.9, 96.8, 102.5, 99.0]),
)
EE_ACTION_DIM = 10


def urdf_joint_limits_deg(urdf_path: str, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    joints = {j.get("name"): j for j in ET.parse(urdf_path).getroot().iter("joint") if j.get("type")}
    lo, hi = [], []
    for name in joint_names:
        limit = joints[name].find("limit")
        lo.append(math.degrees(float(limit.get("lower"))))
        hi.append(math.degrees(float(limit.get("upper"))))
    return np.array(lo), np.array(hi)


def rotation_error_deg(r_target: np.ndarray, r_actual: np.ndarray) -> float:
    cos = (np.trace(r_target.T @ r_actual) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


@dataclass
class CacheJointStateStep(ProcessorStep):
    """Keeps the raw (unnormalized) joint state of the latest observation for the IK initial guess."""

    last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is not None:
            self.last_state = state.detach().to("cpu", torch.float32).clone()
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
class EEToJointsIKStep(ProcessorStep):
    """[x, y, z, r1..r6, gripper] -> [5 arm joints (deg), gripper] with the rules above."""

    cache: CacheJointStateStep
    urdf_path: str = DEFAULT_URDF
    target_frame: str = "gripper_frame_link"
    joint_names: list[str] = field(default_factory=lambda: list(SO101_ARM_JOINTS))
    position_weight: float = 1.0
    orientation_weight: float = 0.01
    # "urdf": placo URDF limits + clip to them; "data": no URDF limits in the solver, clip to the recorded range
    joint_limits: str = "urdf"
    log_path: str | None = None
    _kin: Any = field(default=None, init=False, repr=False)
    _limits: Any = field(default=None, init=False, repr=False)
    _writer: Any = field(default=None, init=False, repr=False)

    def _setup(self):
        from lerobot.model.kinematics import RobotKinematics

        self._kin = RobotKinematics(self.urdf_path, self.target_frame, self.joint_names)
        if self.joint_limits == "urdf":
            self._limits = urdf_joint_limits_deg(self.urdf_path, self.joint_names)
        elif self.joint_limits == "data":
            self._kin.solver.enable_joint_limits(False)
            self._limits = DATA_JOINT_RANGE_DEG
        else:
            raise ValueError(f"joint_limits must be 'urdf' or 'data', got {self.joint_limits!r}")
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            new = not os.path.exists(self.log_path)
            self._file = open(self.log_path, "a", newline="")
            self._writer = csv.writer(self._file)
            if new:
                self._writer.writerow(
                    ["time", "pos_err_mm", "rot_err_deg", "clipped", "fallback"]
                    + [f"q_{n}" for n in self.joint_names]
                )

    def solve(self, ee: np.ndarray, q_now: np.ndarray) -> tuple[np.ndarray, dict]:
        if self._kin is None:
            self._setup()
        target = ee_vector_to_pose(ee[:9].astype(np.float64))
        # RobotKinematics.inverse_kinematics sets the initial joints without update_kinematics(), so its first
        # Newton step would linearize around the previous call's pose. FK at q_now refreshes the model first.
        self._kin.forward_kinematics(q_now.astype(np.float64))
        q = self._kin.inverse_kinematics(
            q_now.astype(np.float64), target,
            position_weight=self.position_weight, orientation_weight=self.orientation_weight,
        )[: len(self.joint_names)]
        fallback = not np.all(np.isfinite(q))
        if fallback:
            q = q_now.copy()
        lo, hi = self._limits
        clipped = bool(np.any(q < lo) or np.any(q > hi))
        q = np.clip(q, lo, hi)
        achieved = self._kin.forward_kinematics(q)
        info = {
            "pos_err_mm": 1000.0 * float(np.linalg.norm(achieved[:3, 3] - target[:3, 3])),
            "rot_err_deg": rotation_error_deg(target[:3, :3], achieved[:3, :3]),
            "clipped": clipped,
            "fallback": fallback,
        }
        return q, info

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if action.shape[-1] != EE_ACTION_DIM:
            raise ValueError(f"EEToJointsIKStep expects {EE_ACTION_DIM}-d EE actions, got {tuple(action.shape)}")
        if self.cache.last_state is None:
            raise RuntimeError("No cached joint state: CacheJointStateStep must run in the preprocessor first")
        ee = action.detach().to("cpu", torch.float32).reshape(-1, EE_ACTION_DIM).numpy()
        q_now = self.cache.last_state.reshape(-1, self.cache.last_state.shape[-1]).numpy()
        out = np.empty((ee.shape[0], len(self.joint_names) + 1), dtype=np.float32)
        for i in range(ee.shape[0]):
            q, info = self.solve(ee[i], q_now[min(i, len(q_now) - 1), : len(self.joint_names)])
            out[i, : len(self.joint_names)] = q
            out[i, -1] = ee[i, 9]  # gripper stays absolute
            if self._writer is not None:
                self._writer.writerow(
                    [f"{time.time():.3f}", f"{info['pos_err_mm']:.3f}", f"{info['rot_err_deg']:.3f}",
                     int(info["clipped"]), int(info["fallback"])] + [f"{v:.3f}" for v in q]
                )
                self._file.flush()
        new_transition = transition.copy()
        shape = (*action.shape[:-1], len(self.joint_names) + 1)
        new_transition[TransitionKey.ACTION] = torch.from_numpy(out).reshape(shape)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def install_ee_ik(
    preprocessor, postprocessor, policy_config, log_path: str | None = None, joint_limits: str = "urdf"
) -> bool:
    """Patch the loaded pipelines in place for an EE checkpoint. Returns False for joint checkpoints."""
    action_ft = policy_config.output_features.get("action")
    if action_ft is None or tuple(action_ft.shape)[-1] != EE_ACTION_DIM:
        return False
    rel = next((s for s in preprocessor.steps if isinstance(s, RelativeEEActionsProcessorStep)), None)
    urdf = rel.urdf_path if rel is not None and rel.urdf_path else DEFAULT_URDF
    cache = CacheJointStateStep()
    idx = next(i for i, s in enumerate(preprocessor.steps) if isinstance(s, AddBatchDimensionProcessorStep))
    preprocessor.steps.insert(idx + 1, cache)
    postprocessor.steps.append(
        EEToJointsIKStep(cache=cache, urdf_path=urdf, log_path=log_path, joint_limits=joint_limits)
    )
    return True
