"""Wrist roll at the grasp vs banana orientation (top-camera image angle at episode start, 0-180, axial).

A parallel gripper grasps the same way at roll and roll+180, and the image angle turns opposite to the roll, so
the training relation is roll + angle = const (mod 180). The constant is fitted on the 180 demonstrations; for each
real run the axial error of the roll commanded at the first close is reported, next to the demonstration spread.
"""
import json
import numpy as np, pandas as pd
from common import EVAL, REPRS, load_all, load_training, gripper_events


def axial(d):
    return ((d + 90) % 180) - 90


starts = {r["ep"]: r for r in json.load(open(EVAL / "train_banana_starts.json"))}
X, Y, Y0 = [], [], []
for e, A, S in load_training():
    c, _ = gripper_events(S[:, 5], A[:, 5])
    if c and e in starts:
        X.append(starts[e]["angle"]); Y.append(A[c[0], 4]); Y0.append(A[0, 4])
X, Y, Y0 = map(np.array, (X, Y, Y0))
offs = np.arange(0, 180, 0.5)
off = offs[np.argmin([np.abs(axial(Y + X - o)).mean() for o in offs])]
tr_err = np.abs(axial(Y + X - off))
print(f"training: roll + angle = {off:.1f} (mod 180); |axial err| median {np.median(tr_err):.1f}, p75 {np.percentile(tr_err, 75):.1f}")
print(f"training: roll change start->close median {np.median(np.abs(Y - Y0)):.1f} deg; |roll at close| spans {Y.min():.0f}..{Y.max():.0f}")
ga = pd.read_csv("grasp_analysis.csv")
rows = []
for r, eps in load_all().items():
    for e in eps:
        g = ga[(ga.repr == r) & (ga.ep == e.ep)].iloc[0]
        c, _ = gripper_events(e.state[:, 5], e.action[:, 5])
        if not c or np.isnan(g.banana_angle):
            continue
        rows.append({"repr": r, "ep": e.ep, "outcome": g.outcome, "banana_angle": g.banana_angle,
                     "roll_close": e.action[c[0], 4], "roll_start": e.action[0, 4],
                     "roll_change": e.action[c[0], 4] - e.action[0, 4],
                     "axial_err": axial(e.action[c[0], 4] + g.banana_angle - off),
                     "err_if_no_turn": axial(e.action[0, 4] + g.banana_angle - off)})
df = pd.DataFrame(rows).round(1)
df.to_csv("wrist_roll.csv", index=False)
df["abs_err"], df["abs_err_no_turn"], df["abs_change"] = df.axial_err.abs(), df.err_if_no_turn.abs(), df.roll_change.abs()
df["ok"] = df.outcome.str.startswith("success")
pd.set_option("display.width", 200)
print(df.groupby("repr")[["abs_err", "abs_err_no_turn", "abs_change"]].median().reindex(REPRS).round(1).to_string())
print(df.groupby(["repr", "ok"]).abs_err.median().unstack().reindex(REPRS).round(1).to_string())
print("share of grasps with |err| < 25 deg:", df.groupby("repr").abs_err.apply(lambda s: round((s < 25).mean() * 100)).reindex(REPRS).to_dict())
print("training share < 25:", round((tr_err < 25).mean() * 100))
