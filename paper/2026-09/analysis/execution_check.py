"""At the decisive gripper close of every episode: did the model aim wrong, or did the robot not get there?

Decisive close = for episodes where the banana was held: the close that starts the holding (measured-commanded
gap >= 6 for >= 1 s); otherwise the close nearest to the first grasp attempt time read from the video
(vreview first_close_s), i.e. the commanded gripper falling below 8. At that moment:
  aim_cm    : commanded gripper position (FK of the command) vs banana (top-camera pixel -> XY map, ~2 cm error)
  reach_cm  : measured position 0.2 s later (the arm's lag) vs banana
  track_cm  : measured vs commanded
  z_cmd/z_meas: height above the demonstrated grasp height
  speed     : measured gripper speed over the 0.5 s before the close (cm/s)
  open_before: widest commanded opening in the 3 s before the close
  roll_err  : wrist-roll misalignment with the banana (axial, from wrist_roll.py's training fit)
  ik_err_mm : IK position residual (EE only)
Joined with the video label (vreview/episode_review.csv)."""
import json
import numpy as np, pandas as pd
from common import EVAL, REPRS, load_all, load_training, fk_pos, gripper_events

starts = {r["ep"]: r for r in json.load(open(EVAL / "train_banana_starts.json"))}
P, XY, Z, OFF = [], [], [], []
for e, A, S in load_training():
    c, _ = gripper_events(S[:, 5], A[:, 5])
    if c and e in starts:
        P.append([starts[e]["x"], starts[e]["y"], 1.0]); XY.append(fk_pos(A[c[0]:c[0] + 1])[0, :2]); Z.append(fk_pos(A[c[0]:c[0] + 1])[0, 2])
M, *_ = np.linalg.lstsq(np.array(P), np.array(XY), rcond=None)
z_grasp = np.median(Z)
axial = lambda d: ((d + 90) % 180) - 90
ROLL_OFF = 22.0  # wrist_roll.py: roll + image angle = 22 (mod 180) in the demonstrations
ga = pd.read_csv("grasp_analysis.csv")
vr = pd.read_csv("vreview/episode_review.csv")
vfull = pd.read_pickle("vreview/all.pkl")


def decisive_close(cmd, meas, video_t):
    held = (meas - cmd >= 6) & (cmd < 10)
    for t in range(len(cmd) - 10):
        if held[t:t + 10].all():
            drops = [k for k in range(max(1, t - 30), t + 1) if cmd[k] < 8 <= cmd[k - 1]]
            return drops[-1] if drops else t
    try:
        tv = int(round(float(str(video_t).split()[0]) * 10))
    except ValueError:
        return None
    drops = [k for k in range(1, len(cmd)) if cmd[k] < 8 <= cmd[k - 1]]
    return min(drops, key=lambda k: abs(k - tv)) if drops else None


rows = []
for r, eps in load_all().items():
    for e in eps:
        g = ga[(ga.repr == r) & (ga.ep == e.ep)].iloc[0]
        v = vr[(vr.repr == r) & (vr.ep == e.ep)].iloc[0]
        vt = vfull[(vfull.repr == r) & (vfull.ep == e.ep)].first_close_s.iloc[0]
        tc = decisive_close(e.action[:, 5], e.state[:, 5], vt)
        row = {"repr": r, "ep": e.ep, "label": v.final_label, "t_close": None if tc is None else tc / 10}
        if tc is not None and isinstance(g.banana_px, str):
            bx, by = map(float, g.banana_px.split(","))
            b = np.array([bx, by, 1.0]) @ M
            pc = fk_pos(e.action[tc:tc + 1])[0]
            tm = min(tc + 2, len(e.state) - 1)
            pm = fk_pos(e.state[tm:tm + 1])[0]
            p0 = fk_pos(e.state[max(0, tm - 5):max(0, tm - 5) + 1])[0]
            row.update(aim_cm=np.linalg.norm(pc[:2] - b) * 100, reach_cm=np.linalg.norm(pm[:2] - b) * 100,
                       track_cm=np.linalg.norm(pm - pc) * 100, z_cmd=(pc[2] - z_grasp) * 100, z_meas=(pm[2] - z_grasp) * 100,
                       speed=np.linalg.norm(pm - p0) * 100 / 0.5, open_before=e.action[max(0, tc - 30):tc, 5].max(),
                       roll_err=abs(axial(e.action[tc, 4] + g.banana_angle - ROLL_OFF)),
                       ik_err_mm=float(e.ik.pos_err_mm.iloc[tc]) if e.ik is not None else np.nan)
        rows.append(row)
df = pd.DataFrame(rows).round(1)
df.to_csv("execution_check.csv", index=False)
pd.set_option("display.width", 220)
df["group"] = df.label.str.replace("success_in", "success").str.replace("success_partial", "success")
cols = ["aim_cm", "reach_cm", "track_cm", "z_cmd", "z_meas", "speed", "open_before", "roll_err"]
print(df.groupby("group")[cols].median().round(1).to_string())
print()
print(df.groupby(["repr", "group"])[cols].median().round(1).to_string())
print()
print(df.sort_values(["repr", "ep"]).to_string(index=False))
