"""Motor discretization (STS3215: 4096 positions/turn) and what it does to the commands of each representation.

lerobot (so_follower, use_degrees=True) converts a joint target in degrees to a Goal_Position tick with
    tick = int(deg * 4095/360 + mid)        (mid = (range_min + range_max)/2 from the calibration; int() truncates)
and a reading back with deg = (tick - mid) * 360/4095; the gripper uses 0-100 over [range_min, range_max].
So arm joints move in 0.0879 deg steps, the gripper in 100/(range) units, and every command is rounded DOWN.
Measured per representation on eval2 (and on the demonstrations for reference):
  1. command quantization error (sent tick vs float command), incl. the truncation bias
  2. per-tick command changes smaller than one tick (the motor target does not change at all)
  3. readback grid: recorded states lie on the tick grid (check) -> relative representations use quantized anchors
  4. static error: measured - sent target while the target is constant (dead band / gravity sag), in ticks
  5. small-motion loss: realized fraction of the commanded motion vs size of the command (per 1.5 s chunk)
Outputs: e2/quant_*.csv
"""
import json
import numpy as np, pandas as pd
from common import REPRS, JOINTS, ARM, load_all, load_training, fk_pos, SESSIONS_EVAL2

CAL = json.load(open("/home/aivlab/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json")) \
    if False else json.load(open(__import__("glob").glob("/home/aivlab/.cache/huggingface/lerobot/calibration/robots/so*_follower/my_follower.json")[0]))
K = 4095 / 360
MID = np.array([(CAL[j]["range_min"] + CAL[j]["range_max"]) / 2 for j in ARM])
GR_MIN, GR_MAX = CAL["gripper"]["range_min"], CAL["gripper"]["range_max"]
STEP = np.r_[np.full(5, 1 / K), 100 / (GR_MAX - GR_MIN)]          # command units per tick


def to_ticks(x):
    """float command (deg / gripper 0-100) -> sent tick, exactly as lerobot does (int() truncation, gripper bounded)."""
    arm = np.floor(x[..., :5] * K + MID).astype(int)                  # all ticks positive -> int() == floor
    g = np.floor(np.clip(x[..., 5], 0, 100) / 100 * (GR_MAX - GR_MIN) + GR_MIN).astype(int)
    return np.concatenate([arm, g[..., None]], -1)


def from_ticks(t):
    arm = (t[..., :5] - MID) / K
    g = (t[..., 5] - GR_MIN) / (GR_MAX - GR_MIN) * 100
    return np.concatenate([arm, g[..., None]], -1)


A = load_all(SESSIONS_EVAL2)
demo = [(None, a, s) for _, a, s in load_training()[::6]]
print("resolution: arm %.4f deg/tick, gripper %.4f units/tick (range %d ticks)" % (1 / K, STEP[5], GR_MAX - GR_MIN))
pos_mm = {}
q0 = np.median(np.concatenate([e.action[:, :5] for e in A["abs_joint"]]), 0)
p0 = fk_pos(q0[None])[0]
for i, j in enumerate(ARM):
    dq = q0.copy(); dq[i] += 1 / K
    pos_mm[j] = np.linalg.norm(fk_pos(dq[None])[0] - p0) * 1000
print("gripper-frame displacement per 1 tick at the median pose (mm):", {k: round(v, 2) for k, v in pos_mm.items()})

rows_q, rows_s, rows_static = [], [], []
for name, eps in list(A.items()) + [("demo", demo)]:
    err, dsub, grid, static = [], [], [], []
    for e in eps:
        act, st = (e.action, e.state) if name != "demo" else (e[1], e[2])
        sent = from_ticks(to_ticks(act))
        err.append(sent - act)
        d = np.abs(np.diff(act, axis=0))
        same_tick = (np.diff(to_ticks(act), axis=0) == 0)
        dsub.append(np.c_[d / STEP, same_tick])
        grid.append(np.abs((st[:, :5] * K + MID) - np.round(st[:, :5] * K + MID)))
        # static: target tick unchanged for >= 6 frames -> measured - sent (ticks) at the last 2 frames of the run
        tt = to_ticks(act); mt = np.c_[np.round(st[:, :5] * K + MID), np.round(st[:, 5] / 100 * (GR_MAX - GR_MIN) + GR_MIN)]
        for jdx in range(6):
            run = 0
            for t in range(1, len(tt)):
                run = run + 1 if tt[t, jdx] == tt[t - 1, jdx] else 0
                if run >= 6 and t + 2 < len(tt):
                    static.append((jdx, mt[t + 2, jdx] - tt[t, jdx]))
    err = np.concatenate(err); dsub = np.concatenate(dsub); grid = np.concatenate(grid)
    st_df = pd.DataFrame(static, columns=["j", "d"])
    for jdx, j in enumerate(JOINTS):
        dj = dsub[:, jdx]
        sd = st_df[st_df.j == jdx].d
        rows_q.append({"set": name, "joint": j, "quant_err_mean": err[:, jdx].mean() / STEP[jdx], "quant_err_absmax": np.abs(err[:, jdx]).max() / STEP[jdx],
                       "step_lt_1tick_%": (dj < 1).mean() * 100, "target_tick_unchanged_%": dsub[:, 6 + jdx].mean() * 100,
                       "median_step_ticks": np.median(dj),
                       "static_err_median_ticks": sd.median() if len(sd) else np.nan, "static_abs_p50": sd.abs().median() if len(sd) else np.nan,
                       "static_abs_p90": sd.abs().quantile(.9) if len(sd) else np.nan, "static_n": len(sd)})
    rows_s.append({"set": name, "state_off_grid_max_ticks": grid.max()})
q = pd.DataFrame(rows_q); q.to_csv("e2/quant_per_joint.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows_s).to_string(index=False))
print(q.round(2).to_string(index=False))

# 5. small-motion loss (per chunk): realized fraction vs commanded chunk travel
c = pd.concat([pd.read_csv(f) for f in __import__("glob").glob("e2/chunks_per_chunk_*.csv")])
c = c[c.chunk > 0]
c["cmd_bin"] = pd.cut(c.cmd_travel_cm, [0, 0.25, 0.5, 1, 2, 5, 50])
t = c.groupby(["cmd_bin"], observed=True).agg(n=("realized", "size"), realized=("realized", "median"), meas=("meas_travel_cm", "median")).round(2)
print(t.to_string())
t.to_csv("e2/quant_small_motion.csv")
print(c.groupby(["repr", "cmd_bin"], observed=True).realized.median().unstack().round(2).reindex(REPRS).to_string())
