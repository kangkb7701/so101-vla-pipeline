"""
Shared types + heuristic helpers for supervisor implementations.

This module is dependency-light (only numpy + scipy) so both GeminiSupervisor and
QwenSupervisor can import it, and main_real2.py can use the dataclasses without
pulling in google-genai or transformers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
# Constants
# ============================================================

STATUS_VALUES = (
    'in_progress',
    'subgoal_complete',
    'corrective_action',
    'stuck',
    'failed',
    'task_complete',
)
ALLOW_SWITCH_STATUSES = frozenset({
    'subgoal_complete',
    'corrective_action',
    'stuck',
})


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class StateSnapshot:
    """One sample of robot telemetry, sampled at ~5Hz by the main control thread."""
    t: float                    # monotonic timestamp (s)
    instruction: str            # current instruction at sample time
    q_deg: list                 # 6 arm joints, degrees
    gripper_val: float          # gripper position (deg or raw scale, consistent unit)
    ee_pos: list                # [x, y, z] in meters
    ee_rpy: list                # [roll, pitch, yaw] in degrees


@dataclass
class ObjectiveSignals:
    """Hard-coded heuristics computed from recent StateSnapshots."""
    gripper_closed_recent: bool   # gripper_val > threshold for ≥4 of last 5 samples
    ee_z_delta_recent: float      # ee_z(now) - ee_z(~2s ago), meters, signed
    motion_magnitude: float       # mean ||Δq_arm||_2 / dt over last ~1s, deg/s


@dataclass
class InstructionUseRecord:
    """One completed entry in the episode-level instruction history."""
    instruction: str
    started_at_monotonic: float
    duration_s: float


@dataclass
class SupervisorDecision:
    """One supervisor call result. Always returned by monitor() (no exceptions)."""
    status: str                 # one of STATUS_VALUES
    instruction: str            # next instruction (or current, if not switching)
    reason: str                 # <=1 sentence
    latency_ms: float
    raw_response: str           # for debug logs
    phase: str = ''             # approach|align|pre_grasp|grasp|lift|complete|recover


# ============================================================
# Helpers
# ============================================================

def rot_mat_to_rpy_deg(R) -> list:
    """3x3 rotation matrix → [roll, pitch, yaw] degrees (XYZ extrinsic).

    Uses scipy for gimbal-lock robustness.
    """
    return Rotation.from_matrix(np.asarray(R)).as_euler('xyz', degrees=True).tolist()


def compute_signals(history: list, thresholds: dict) -> ObjectiveSignals:
    """Compute objective heuristics from recent StateSnapshot history.

    Args:
        history: oldest-first list of StateSnapshot, sampled at ~5Hz.
        thresholds: dict with optional key 'gripper_closed' (default 25.0).

    Returns:
        ObjectiveSignals with safe defaults if history is too short.
    """
    n = len(history)
    if n < 2:
        print(f"⚠️ [Supervisor] history too short (n={n}), returning safe defaults")
        return ObjectiveSignals(False, 0.0, 0.0)

    closed_thresh = thresholds.get('gripper_closed', 25.0)

    if n >= 5:
        last5 = history[-5:]
        n_closed = sum(1 for s in last5 if s.gripper_val > closed_thresh)
        gripper_closed_recent = n_closed >= 4
    else:
        gripper_closed_recent = False

    target_idx = max(0, n - 11)
    ee_z_delta_recent = float(history[-1].ee_pos[2] - history[target_idx].ee_pos[2])

    window = history[-min(n, 5):]
    vals = []
    for i in range(1, len(window)):
        dq = np.array(window[i].q_deg) - np.array(window[i - 1].q_deg)
        dt = window[i].t - window[i - 1].t
        if dt > 1e-6:
            vals.append(float(np.linalg.norm(dq) / dt))
    motion_magnitude = float(np.mean(vals)) if vals else 0.0

    return ObjectiveSignals(
        gripper_closed_recent=gripper_closed_recent,
        ee_z_delta_recent=ee_z_delta_recent,
        motion_magnitude=motion_magnitude,
    )
