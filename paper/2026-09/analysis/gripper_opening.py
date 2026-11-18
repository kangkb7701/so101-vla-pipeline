"""How wide was the gripper opened before the first close, and how low was the gripper at that moment?"""
import numpy as np, pandas as pd
from common import REPRS, load_all, load_training, fk_pos, gripper_events
vis = pd.read_csv("outcomes_visual.csv")
tr = load_training()
op, zz = [], []
for e, A, S in tr:
    c, _ = gripper_events(S[:, 5], A[:, 5])
    if c:
        op.append(A[max(0, c[0] - 30):c[0], 5].max()); zz.append(fk_pos(A[c[0]:c[0]+1])[0, 2])
z_grasp = np.median(zz)
print(f"training: gripper opening before grasp median {np.median(op):.1f} (p10 {np.percentile(op,10):.1f}, p90 {np.percentile(op,90):.1f}) | grasp height {z_grasp*100:.1f} cm")
rows = []
for r, eps in load_all().items():
    for e in eps:
        c, _ = gripper_events(e.state[:, 5], e.action[:, 5])
        if not c:
            continue
        t = c[0]
        rows.append({"repr": r, "ep": e.ep, "outcome": vis[(vis.repr == r) & (vis.ep == e.ep)].outcome.iloc[0],
                     "open_before_close": e.action[max(0, t - 30):t, 5].max(), "meas_open_before_close": e.state[max(0, t - 30):t, 5].max(),
                     "height_at_close_cm": (fk_pos(e.action[t:t+1])[0, 2] - z_grasp) * 100})
df = pd.DataFrame(rows)
df["succ"] = df.outcome.str.startswith("success")
print(df.groupby("repr")[["open_before_close", "meas_open_before_close", "height_at_close_cm"]].median().round(1).reindex(REPRS).to_string())
print(df.groupby(["repr", "succ"])[["open_before_close"]].median().round(1).unstack().reindex(REPRS).to_string())
print("share of first closes with opening < 20:", df.groupby("repr").apply(lambda d: round((d.open_before_close < 20).mean(), 2), include_groups=False).reindex(REPRS).to_dict())
df.to_csv("gripper_opening.csv", index=False)
