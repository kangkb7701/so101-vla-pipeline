"""Why did each policy act as it did? Counterfactual replays on the recorded observations (replay_lib.Replayer).

For every episode three decision points are re-run with 5 noise seeds, first unchanged and then with one input
changed (images always as recorded):

1. Start (chunk at frame 0): does the arm leave the rest pose?
   base | grip37: gripper state set to 37 (all demonstrations start open ~37) | trainstart: whole joint state set
   to the median demonstration start pose (pan -5.7 instead of about -18, gripper 37).
   Measure: displacement of the chunk-end gripper position from the (possibly modified) start position.
2. Decisive close (chunk that contained the decisive close, execution_check.csv):
   base | grip37 | roll+30 / roll-30: wrist_roll state shifted by 30 deg.
   Measure: does the chunk command a close (gripper < 8), chunk-end distance to the banana, and how much the
   commanded roll follows the roll state (copy gain = change of commanded roll / 30).
3. After the close (next chunk): base | grip11 (measured gripper = 11, the value while holding the banana) |
   grip0 (= empty jaws). Measure: distance of the chunk end to the target basket and to the nearest other
   basket, and whether the gripper reopens (> 15) — i.e. a retry.
"""
import sys, json
import numpy as np, pandas as pd
from replay_lib import Replayer
from common import EVAL, REPRS, load_all, load_training, fk_pos, gripper_events

SEEDS = range(5)
TRAIN_START = np.array([-5.7, -103.0, 95.5, 79.2, -43.6, 37.2], dtype=np.float32)
BASKET = {k: np.array(v) for k, v in json.load(open("basket_release_xyz.json")).items()}

starts = {r["ep"]: r for r in json.load(open(EVAL / "train_banana_starts.json"))}
P, XY = [], []
for e, A, S in load_training():
    c, _ = gripper_events(S[:, 5], A[:, 5])
    if c and e in starts:
        P.append([starts[e]["x"], starts[e]["y"], 1.0]); XY.append(fk_pos(A[c[0]:c[0] + 1])[0, :2])
M, *_ = np.linalg.lstsq(np.array(P), np.array(XY), rcond=None)
ga = pd.read_csv("grasp_analysis.csv")
ex = pd.read_csv("execution_check.csv")
vr = pd.read_csv("vreview/episode_review.csv")


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


if __name__ == "__main__":
    A = load_all()
    reprs = sys.argv[1:] or REPRS
    rows = []
    for r in reprs:
        R = Replayer(r)
        for e in A[r]:
            g = ga[(ga.repr == r) & (ga.ep == e.ep)].iloc[0]
            banana = (np.array([*map(float, g.banana_px.split(",")), 1.0]) @ M) if isinstance(g.banana_px, str) else None
            v = vr[(vr.repr == r) & (vr.ep == e.ep)].iloc[0]
            target = BASKET[v.target_color]
            others = [BASKET[c] for c in BASKET if c != v.target_color]
            base = dict(repr=r, ep=e.ep, label=v.final_label)

            # 1. start
            for cond, mod, s0 in (("base", None, e.state[0]), ("grip37", set_state(5, 37.0), np.r_[e.state[0, :5], 37.0]),
                                  ("trainstart", whole_state(TRAIN_START), TRAIN_START.astype(np.float64))):
                p0 = fk_pos(s0[None])[0]
                for s in SEEDS:
                    ch = R.chunk_at(e.ep, 0, seed=s, modify=mod)
                    rows.append({**base, "point": "start", "cond": cond, "seed": s,
                                 "disp_cm": np.linalg.norm(fk_pos(ch[-1:])[0] - p0) * 100,
                                 "grip_max": ch[:, 5].max(), "grip_min": ch[:, 5].min()})

            x = ex[(ex.repr == r) & (ex.ep == e.ep)].iloc[0]
            if np.isnan(x.t_close):
                continue
            tc = int(round(x.t_close * 10)); b = 15 * (tc // 15)
            # 2. decisive close
            for cond, mod in (("base", None), ("grip37", set_state(5, 37.0)), ("roll+30", add_state(4, 30.0)), ("roll-30", add_state(4, -30.0))):
                for s in SEEDS:
                    ch = R.chunk_at(e.ep, b, seed=s, modify=mod)
                    pe = fk_pos(ch[-1:])[0]
                    rows.append({**base, "point": "close", "cond": cond, "seed": s, "t": b / 10,
                                 "closes": bool((ch[:, 5] < 8).any()), "grip_min": ch[:, 5].min(), "grip_max": ch[:, 5].max(),
                                 "roll_end": ch[-1, 4], "z_end_cm": pe[2] * 100,
                                 "banana_cm": np.linalg.norm(pe[:2] - banana[:2]) * 100 if banana is not None else np.nan})
            # 3. after the close
            b2 = b + 15
            if b2 < len(e.action):
                for cond, mod in (("base", None), ("grip11", set_state(5, 11.0)), ("grip0", set_state(5, 0.0))):
                    for s in SEEDS:
                        ch = R.chunk_at(e.ep, b2, seed=s, modify=mod)
                        pe = fk_pos(ch[-1:])[0]
                        rows.append({**base, "point": "after", "cond": cond, "seed": s, "t": b2 / 10,
                                     "target_cm": np.linalg.norm(pe[:2] - target[:2]) * 100,
                                     "other_cm": min(np.linalg.norm(pe[:2] - o[:2]) for o in others) * 100,
                                     "banana_cm": np.linalg.norm(pe[:2] - banana[:2]) * 100 if banana is not None else np.nan,
                                     "reopens": bool((ch[:, 5] > 15).any()), "z_end_cm": pe[2] * 100,
                                     "meas_gap": float(e.state[b2, 5] - e.action[b2, 5])})
            print(r, e.ep, "done", flush=True)
        R.close()
    pd.DataFrame(rows).to_csv(f"counterfactual_{'_'.join(reprs)}.csv", index=False)
