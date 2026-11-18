"""EE runs: how well did IK realise the policy's EE targets? (IK log: target-vs-achieved error, clip/fallback flags)

- position / orientation residual (target pose vs FK of the IK solution) over each episode and at grasp events
- URDF joint-limit saturation of the IK solution (within 0.5 deg of a limit) and of the measured start pose
- residual vs pose: does the residual grow near the URDF limits or at the rest pose?
"""
import numpy as np, pandas as pd
from common import URDF, ARM, LABEL, COLOR, INK2, MUTED, load_all, gripper_events, setup_matplotlib
import sys; sys.path.insert(0, "/home/aivlab/kkb_capstone/2026-09-paper/eval")
from ee_ik_steps import urdf_joint_limits_deg

lo, hi = urdf_joint_limits_deg(URDF, ARM)
A = load_all()
rows, series = [], {}
for r in ("abs_ee", "rel_ee"):
    for e in A[r]:
        ik = e.ik
        q = ik[[f"q_{j}" for j in ARM]].to_numpy()
        sat = (np.abs(q - lo) < 0.5) | (np.abs(q - hi) < 0.5)
        start_out = (e.state[0, :5] < lo) | (e.state[0, :5] > hi)
        closes, _ = gripper_events(e.state[:, 5], e.action[:, 5])
        pe, re = ik.pos_err_mm.to_numpy(), ik.rot_err_deg.to_numpy()
        rows.append({
            "repr": r, "ep": e.ep, "pos_err_mm_p50": round(np.median(pe), 1), "pos_err_mm_p95": round(np.percentile(pe, 95), 1),
            "pos_err_mm_max": round(pe.max(), 1), "rot_err_deg_p50": round(np.median(re), 1), "rot_err_deg_p95": round(np.percentile(re, 95), 1),
            "rot_err_deg_max": round(re.max(), 1),
            "frames_saturated_%": round(sat.any(1).mean() * 100, 1),
            "saturated_joints": ",".join(j for j, s in zip(ARM, sat.any(0)) if s),
            "start_outside_urdf": ",".join(j for j, s in zip(ARM, start_out) if s),
            "pos_err_first_2s_mm": round(pe[:20].mean(), 1), "pos_err_after_2s_mm": round(pe[20:].mean(), 1),
            "pos_err_at_close_mm": round(pe[closes[0]], 1) if closes else None,
            "clipped": int(ik.clipped.sum()), "fallback": int(ik.fallback.sum()),
        })
        series[(r, e.ep)] = (pe, re, sat.any(1))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("URDF limits (deg):", dict(zip(ARM, zip(np.round(lo, 1), np.round(hi, 1)))))
print(df.to_string(index=False))
df.to_csv("ik_analysis.csv", index=False)
for r in ("abs_ee", "rel_ee"):
    pe = np.concatenate([series[k][0] for k in series if k[0] == r]); re = np.concatenate([series[k][1] for k in series if k[0] == r])
    sat = np.concatenate([series[k][2] for k in series if k[0] == r])
    print(f"{r}: pos err mm p50 {np.median(pe):.1f} p95 {np.percentile(pe,95):.1f} | rot err deg p50 {np.median(re):.1f} p95 {np.percentile(re,95):.1f}"
          f" | saturated frames {sat.mean()*100:.1f}% | pos err when saturated {pe[sat].mean() if sat.any() else float('nan'):.1f} vs not {pe[~sat].mean():.1f}")

plt = setup_matplotlib()
fig, ax = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
for (r, ep), (pe, re, sat) in series.items():
    t = np.arange(len(pe)) / 10
    ax[0].plot(t, pe, color=COLOR[r], lw=1, alpha=0.8)
    ax[1].plot(t, re, color=COLOR[r], lw=1, alpha=0.8)
ax[0].set_ylabel("위치 오차 (mm)"); ax[1].set_ylabel("방향 오차 (°)"); ax[1].set_xlabel("시간 (s)")
ax[0].set_title("역기구학: 모델 목표 자세와 실제로 만든 자세의 차이 (에피소드별 선)", fontsize=10, color=INK2, loc="left")
for r in ("abs_ee", "rel_ee"):
    ax[0].plot([], [], color=COLOR[r], label=LABEL[r])
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figs/ik_residual.png", dpi=150)
