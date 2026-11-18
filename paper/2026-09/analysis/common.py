"""Shared loaders for the real-robot evaluation analysis (2026-09-paper).

Only data that was actually recorded is used: per-frame commanded joints (`action`, what lerobot-rollout sent),
measured joints (`observation.state`), camera frames, and for EE runs the IK log rows (IK output joints, target-vs-
achieved pose error, clip/fallback flags). Raw policy outputs (EE targets, relative offsets) were not logged.
Run scripts from 2026-09-paper/analysis (the repo root has a `datasets/` package that shadows HF datasets).
"""

import csv
import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path("/home/aivlab/kkb_capstone/2026-09-paper")
EVAL = PAPER / "eval"
REC = EVAL / "recordings"
TRAIN_ROOT = Path("/home/aivlab/kkb_capstone/data/lerobot/so101_pick_place_fruit_180")
URDF = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS = ARM + ["gripper"]
REPRS = ["abs_joint", "rel_joint", "abs_ee", "rel_ee"]
LABEL = {"abs_joint": "절대 관절각", "rel_joint": "관절 상대", "abs_ee": "절대 EE", "rel_ee": "상대 EE"}
# dataviz reference palette, categorical slots 1-4 in fixed order (one color per representation)
COLOR = {"abs_joint": "#2a78d6", "rel_joint": "#eb6834", "abs_ee": "#1baf7a", "rel_ee": "#eda100"}
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
FPS = 10


@dataclass
class Episode:
    repr: str
    session: str          # eval / eval1
    ep: int               # episode index inside the session dataset
    trial: int | None
    plan: dict = field(default_factory=dict)
    action: np.ndarray = None   # (T, 6) commanded joint targets sent to the robot (deg; gripper 0-100)
    state: np.ndarray = None    # (T, 6) measured joints
    ik: pd.DataFrame | None = None  # IK rows aligned to frames (EE only)

    @property
    def name(self):
        return f"{self.repr}/{self.session}/ep{self.ep}" + (f" (trial {self.trial})" if self.trial else "")


def read_parquets(root, cols):
    files = sorted(glob.glob(str(Path(root) / "data" / "**" / "*.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def plan_rows(session):
    name = "plan.csv" if session == "eval" else f"plan_{session}.csv"
    with open(EVAL / "eval_plan" / name) as f:
        return {int(r["trial"]): r for r in csv.DictReader(f)}


def trial_map(repr_, session):
    """episode index -> trial number, from the session's results sheet (written by run_eval_session.sh)."""
    name = f"results_{repr_}.csv" if session == "eval" else f"results_{repr_}_{session}.csv"
    p = EVAL / "eval_plan" / name
    out = {}
    if p.exists():
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get("trial") and r["episode_index"] not in ("", "none", None):  # skips blank lines
                    out[int(r["episode_index"])] = int(r["trial"])  # later rows (re-runs) win
    return out


def align_ik(ik_all: pd.DataFrame, action: np.ndarray, start_hint: int):
    """Find the IK rows whose output joints equal this episode's commanded joints (exact float match)."""
    q = ik_all[[f"q_{j}" for j in ARM]].to_numpy()
    a = action[:, :5]
    T = len(a)
    best = None
    for s in range(max(0, start_hint - 5), len(q) - T + 1):
        if np.abs(q[s] - a[0]).max() < 0.01 and np.abs(q[s + T - 1] - a[-1]).max() < 0.01:
            if np.abs(q[s:s + T] - a).max() < 0.01:
                best = s
                break
    if best is None:
        return None, start_hint
    return ik_all.iloc[best:best + T].reset_index(drop=True), best + T


def load_session(repr_, session):
    root = REC / f"smolvla_{repr_}" / session
    df = read_parquets(root, ["action", "observation.state", "episode_index", "frame_index"])
    tmap = trial_map(repr_, session)
    plans = plan_rows(session) if (EVAL / "eval_plan" / ("plan.csv" if session == "eval" else f"plan_{session}.csv")).exists() else {}
    ik_all = None
    ik_path = REC / f"smolvla_{repr_}" / f"{session}.ik.csv"
    if ik_path.exists():
        ik_all = pd.read_csv(ik_path)
    eps, cursor = [], 0
    for e, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        trial = tmap.get(int(e))
        epi = Episode(repr_, session, int(e), trial, plans.get(trial, {}) if trial else {},
                      np.stack(g["action"]).astype(np.float64), np.stack(g["observation.state"]).astype(np.float64))
        if ik_all is not None:
            epi.ik, cursor = align_ik(ik_all, epi.action, cursor)
        eps.append(epi)
    return eps


SESSIONS_OLD = {"abs_joint": "eval", "rel_joint": "eval", "abs_ee": "eval1", "rel_ee": "eval1"}
SESSIONS_EVAL2 = {r: "eval2" for r in REPRS}


def load_all(sessions=None):
    """Default: the first evaluation; environment EVAL_SESSIONS=eval2 switches every script to the eval2 sessions."""
    import os
    sessions = sessions or (SESSIONS_EVAL2 if os.environ.get("EVAL_SESSIONS") == "eval2" else SESSIONS_OLD)
    return {r: load_session(r, s) for r, s in sessions.items()}


def load_training():
    df = read_parquets(TRAIN_ROOT, ["action", "observation.state", "episode_index", "frame_index"])
    out = []
    for e, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        out.append((int(e), np.stack(g["action"]).astype(np.float64), np.stack(g["observation.state"]).astype(np.float64)))
    return out


_KIN = None


def kin():
    global _KIN
    if _KIN is None:
        from lerobot.model.kinematics import RobotKinematics
        _KIN = RobotKinematics(URDF, "gripper_frame_link", ARM)
    return _KIN


def fk_pos(joints_deg: np.ndarray) -> np.ndarray:
    """(T, >=5) joints -> (T, 3) gripper-frame position in metres (robot base frame)."""
    k = kin()
    return np.array([k.forward_kinematics(j[:5])[:3, 3] for j in joints_deg])


def fk_T(joints_deg: np.ndarray) -> np.ndarray:
    k = kin()
    return np.array([k.forward_kinematics(j[:5]) for j in joints_deg])


def gripper_events(state_grip: np.ndarray, action_grip: np.ndarray):
    """Close event = commanded gripper falls below 8 after having been opened to >= 15 since the last close;
    open event = commanded gripper rises above 15 after a close. Uses the commanded signal (policy intent)."""
    closes, opens = [], []
    peak, closed = 0.0, False
    for t, g in enumerate(action_grip):
        if not closed:
            peak = max(peak, g)
            if g < 8 and peak >= 15:
                closes.append(t)
                closed = True
        elif g > 15:
            opens.append(t)
            closed, peak = False, g
    return closes, opens


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ["NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK TC", "Noto Sans KR", "Noto Serif CJK KR"]:
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "text.color": INK,
    })
    return plt
