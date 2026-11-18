"""Per-episode overview for every session (eval / eval1 / eval2): outcome (video review), start pose, banana
placement, timing, gripper, wrist, aim at the decisive close. Output: e2/overview.csv + printed summaries."""
import re
import numpy as np, pandas as pd
from common import REPRS, load_session, fk_pos, gripper_events
from e2_common import banana_info, banana_xy, decisive_close, BASKET, Z_GRASP

ax = lambda d: ((d + 90) % 180) - 90
first = lambda s: re.split(r"[ (;,:]", str(s).strip())[0]
V_OLD = pd.read_csv("vreview/episode_review.csv")
V_NEW = pd.read_pickle("vreview_eval2/all.pkl")


def label(r, s, ep):
    if s == "eval2":
        v = V_NEW[(V_NEW.repr == r) & (V_NEW.ep == ep)].iloc[0]
        o, f = v.outcome_k, v.failure_point_k
        return "success" if o == "success" else ("rim" if o == "rim" else f"fail_{f}")
    lab = V_OLD[(V_OLD.repr == r) & (V_OLD.ep == ep)].final_label.iloc[0]
    return {"success_in": "success", "success_partial": "rim"}.get(lab, lab)


def reversals(x, amp=4.0):
    n, last, d = 0, x[0], 0
    for v in x[1:]:
        if d >= 0 and v - last < -amp: n += d > 0; d = -1; last = v
        elif d <= 0 and v - last > amp: n += d < 0; d = 1; last = v
        elif (d > 0 and v > last) or (d < 0 and v < last): last = v
    return n


rows = []
for r in REPRS:
    for s in (("eval", "eval2") if "joint" in r else ("eval1", "eval2")):
        for e in load_session(r, s):
            a, st = e.action, e.state
            pm = fk_pos(st); pc = fk_pos(a)
            d0 = np.linalg.norm(pm - pm[0], axis=1) * 100
            bp = banana_info(r, e.ep, s); b = banana_xy(r, e.ep, s)
            tc, th = decisive_close(a, st)
            row = {"repr": r, "session": s, "ep": e.ep, "trial": e.trial, "orient_plan": e.plan.get("orientation"),
                   "label": label(r, s, e.ep), "start_pan": st[0, 0], "start_roll": st[0, 4], "start_grip": st[0, 5],
                   "len_s": len(a) / 10, "leave_s": np.argmax(d0 > 5) / 10 if d0.max() > 5 else np.nan,
                   "jitter_per_min": (reversals(a[:, 1]) + reversals(a[:, 2])) / (len(a) / 600),
                   "grip_max_before_close": a[:tc, 5].max() if tc else a[:, 5].max(), "held": th is not None,
                   "banana_px_x": bp[0] if bp else np.nan, "banana_px_y": bp[1] if bp else np.nan,
                   "banana_angle": bp[2] if bp else np.nan}
            if tc is not None:
                row.update(t_close=tc / 10, z_close_cm=(pc[tc, 2] - Z_GRASP) * 100, roll_change=a[tc, 4] - a[0, 4],
                           roll_close=a[tc, 4])
                if b is not None:
                    row.update(aim_cm=np.linalg.norm(pc[tc, :2] - b) * 100, roll_err=abs(ax(a[tc, 4] + bp[2] - 22.0)),
                               roll_err_if_no_turn=abs(ax(a[0, 4] + bp[2] - 22.0)))
            rows.append(row)
df = pd.DataFrame(rows)
df["ok"] = df.label.isin(["success", "rim"])
df.to_csv("e2/overview.csv", index=False)
if __name__ == "__main__":
    pd.set_option("display.width", 250)
    g = df.groupby(["repr", "session"])
    print(pd.DataFrame({"n": g.size(), "success": g.label.apply(lambda x: (x == "success").sum()), "rim": g.label.apply(lambda x: (x == "rim").sum()),
                        "held": g.held.sum()}).reindex(REPRS, level=0).to_string())
    cols = ["start_pan", "start_roll", "start_grip", "leave_s", "grip_max_before_close", "z_close_cm", "aim_cm", "roll_err",
            "roll_err_if_no_turn", "roll_change", "jitter_per_min", "len_s"]
    print(g[cols].median().round(1).reindex(REPRS, level=0).T.to_string())
    print(pd.crosstab([df.repr, df.session], df.label.str.replace("fail_", "")).reindex(REPRS, level=0).to_string())
