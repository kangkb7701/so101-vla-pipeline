"""Sensitivity analysis on the eval2 recordings (replay_lib: recorded observations, same pipeline as on the robot).

A. Chunk that contained the decisive gripper close of each episode (e2_common.decisive_close), 5 noise seeds:
   base | grip37: gripper state set to 37 (all demonstrations start with the gripper at ~37) |
   roll+30 / roll-30: wrist_roll state shifted by 30 deg.
   -> share of seeds commanding a close (gripper < 8), commanded roll at chunk end (gain = change / 30),
      chunk-end distance to the banana.
B. Start-up: chunks predicted at 0 / 3 / 6 / 9 s while the gripper is still within 5 cm of its start position,
   10 seeds: base | grip37 | trainstart (whole joint state = median demonstration start pose).
   -> p_go = share of chunks whose end moves the gripper > 3 cm.
Outputs: e2/sens_close.csv, e2/sens_start.csv
"""
import sys
import numpy as np, pandas as pd
import replay_lib
from replay_lib import Replayer
from common import REPRS, load_session, fk_pos
from e2_common import banana_xy, decisive_close

TRAIN_START = np.array([-5.7, -103.0, 95.5, 79.2, -43.6, 37.2], dtype=np.float32)
OV = pd.read_csv("e2/overview.csv")


def set_state(idx, val):
    def f(t, o):
        s = o["observation.state"].copy(); s[idx] = val; o["observation.state"] = s; return o
    return f


def add_state(idx, d):
    def f(t, o):
        s = o["observation.state"].copy(); s[idx] += d; o["observation.state"] = s; return o
    return f


def whole_state(v):
    def f(t, o):
        o["observation.state"] = v.copy(); return o
    return f


rows_c, rows_s = [], []
for r in (sys.argv[1:] or REPRS):
    replay_lib.SESSION[r] = "eval2"
    R = Replayer(r)
    for e in load_session(r, "eval2"):
        label = OV[(OV.repr == r) & (OV.session == "eval2") & (OV.ep == e.ep)].label.iloc[0]
        b = banana_xy(r, e.ep)
        tc, th = decisive_close(e.action, e.state)
        if tc is not None:
            t0 = 15 * (tc // 15)
            for cond, mod in (("base", None), ("grip37", set_state(5, 37.0)), ("roll+30", add_state(4, 30.0)),
                              ("roll-30", add_state(4, -30.0))):
                for sd in range(5):
                    ch = R.chunk_at(e.ep, t0, seed=sd, modify=mod)
                    pe = fk_pos(ch[-1:])[0]
                    rows_c.append({"repr": r, "ep": e.ep, "label": label, "held": th is not None, "cond": cond, "seed": sd,
                                   "t": t0 / 10, "closes": bool((ch[:, 5] < 8).any()), "grip_max": ch[:, 5].max(),
                                   "roll_end": ch[-1, 4], "roll_state": e.state[t0, 4],
                                   "banana_cm": np.linalg.norm(pe[:2] - b[:2]) * 100 if b is not None else np.nan})
        pm = fk_pos(e.state); d0 = np.linalg.norm(pm - pm[0], axis=1) * 100
        for t in (0, 30, 60, 90):
            if t >= len(e.state) or d0[:t + 1].max() > 5:
                continue
            for cond, mod, sp in (("base", None, e.state[t]), ("grip37", set_state(5, 37.0), np.r_[e.state[t, :5], 37.0]),
                                  ("trainstart", whole_state(TRAIN_START), TRAIN_START.astype(float))):
                p0 = fk_pos(np.asarray(sp, dtype=float)[None])[0]
                for sd in range(10):
                    ch = R.chunk_at(e.ep, t, seed=sd, modify=mod)
                    rows_s.append({"repr": r, "ep": e.ep, "label": label, "t": t / 10, "cond": cond, "seed": sd,
                                   "disp_cm": np.linalg.norm(fk_pos(ch[-1:])[0] - p0) * 100})
        print(r, e.ep, flush=True)
    R.close()
tag = "_".join(sys.argv[1:]) or "all"
pd.DataFrame(rows_c).to_csv(f"e2/sens_close_{tag}.csv", index=False)
pd.DataFrame(rows_s).to_csv(f"e2/sens_start_{tag}.csv", index=False)
