"""rel_ee green episodes: what happened during the large up-down oscillation?

For each EE episode: oscillation score (direction reversals of shoulder_lift/elbow commands with > 4 deg swings),
IK position residual, reach (distance of the gripper frame from the shoulder), position-Jacobian conditioning
(smallest singular value / condition number of d(position)/d(joints) at the commanded joints), URDF saturation,
and whether residual spikes line up with the 15-frame re-planning boundary.
"""
import numpy as np, pandas as pd
from common import URDF, ARM, LABEL, COLOR, INK2, MUTED, load_all, kin, fk_pos, setup_matplotlib
import sys; sys.path.insert(0, "/home/aivlab/kkb_capstone/2026-09-paper/eval")
from ee_ik_steps import urdf_joint_limits_deg

lo, hi = urdf_joint_limits_deg(URDF, ARM)
k = kin()
SHOULDER = k.forward_kinematics(np.zeros(5))  # placeholder, replaced below


def jac_pos(q_deg, h=0.5):
    p0 = k.forward_kinematics(q_deg)[:3, 3]
    J = np.zeros((3, 5))
    for i in range(5):
        dq = q_deg.copy(); dq[i] += h
        J[:, i] = (k.forward_kinematics(dq)[:3, 3] - p0) / np.radians(h)
    s = np.linalg.svd(J[:, :4], compute_uv=False)  # wrist_roll does not move the gripper-frame origin much
    return s.min(), s.max() / max(s.min(), 1e-9)


def reversals(x, amp=4.0):
    """number of direction reversals whose swing exceeds amp degrees"""
    n, last_ext, direction = 0, x[0], 0
    for v in x[1:]:
        d = v - last_ext
        if direction >= 0 and d < -amp:
            n += direction > 0; direction = -1; last_ext = v
        elif direction <= 0 and d > amp:
            n += direction < 0; direction = 1; last_ext = v
        elif (direction > 0 and v > last_ext) or (direction < 0 and v < last_ext):
            last_ext = v
    return n


A = load_all()
# reach reference: max distance of gripper frame from base over the training commands
from common import load_training
tr_pos = np.concatenate([fk_pos(a[::5]) for _, a, _ in load_training()[:60]])
reach_train = np.linalg.norm(tr_pos[:, :2], axis=1)
print(f"training horizontal reach (cm): p50 {np.median(reach_train)*100:.1f} p99 {np.percentile(reach_train,99)*100:.1f} max {reach_train.max()*100:.1f}")

rows, detail = [], {}
for r in ("abs_ee", "rel_ee"):
    for e in A[r]:
        pe = e.ik.pos_err_mm.to_numpy()
        q = e.action[:, :5]
        smin = np.array([jac_pos(qq)[0] for qq in q]); cond = np.array([jac_pos(qq)[1] for qq in q])
        pos = fk_pos(e.action); reach = np.linalg.norm(pos[:, :2], axis=1)
        osc = reversals(e.action[:, 1]) + reversals(e.action[:, 2])
        spike = pe > 20
        sat = ((np.abs(q - lo) < 0.5) | (np.abs(q - hi) < 0.5)).any(1)
        rows.append({"repr": r, "ep": e.ep, "color": e.plan.get("color", ""), "oscillation_reversals": osc,
                     "ik_err_max_mm": round(pe.max(), 1), "frames_ik_err>20mm": int(spike.sum()),
                     "reach_max_cm": round(reach.max() * 100, 1), "smin_min": round(smin.min(), 3),
                     "cond_max": round(cond.max(), 1), "smin_at_spikes": round(smin[spike].mean(), 3) if spike.any() else None,
                     "reach_at_spikes_cm": round(reach[spike].mean() * 100, 1) if spike.any() else None,
                     "sat_at_spikes_%": round(sat[spike].mean() * 100) if spike.any() else None})
        detail[(r, e.ep)] = dict(pe=pe, q=q, meas=e.state[:, :5], smin=smin, reach=reach, pos=pos, sat=sat, grip=e.action[:, 5])
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.to_string(index=False))
df.to_csv("oscillation.csv", index=False)

# spike timing vs re-planning boundary (rel_ee boundary phase from tracking.py = 6)
for key in [(r, ep) for r, ep in detail if detail[(r, ep)]["pe"].max() > 50]:
    d = detail[key]
    on = np.where(np.diff((d["pe"] > 20).astype(int)) == 1)[0] + 1
    print(key, "IK residual spike onsets at frames", on.tolist(), "-> mod 15 =", (on % 15).tolist())
    t0 = on[0]
    for t in range(max(0, t0 - 3), min(len(d["pe"]), t0 + 25)):
        print(f"  t={t:3d} ({t/10:5.1f}s) ik_err {d['pe'][t]:6.1f}mm  cmd lift {d['q'][t,1]:7.1f} elbow {d['q'][t,2]:7.1f} wflex {d['q'][t,3]:6.1f}"
              f" | meas lift {d['meas'][t,1]:7.1f} elbow {d['meas'][t,2]:7.1f} | reach {d['reach'][t]*100:5.1f}cm z {d['pos'][t,2]*100:5.1f}cm"
              f" | smin {d['smin'][t]:.3f} sat {int(d['sat'][t])} grip {d['grip'][t]:5.1f}")

plt = setup_matplotlib()
keys = [(r, ep) for r, ep in detail if detail[(r, ep)]["pe"].max() > 50] + [("rel_ee", 5)]
fig, axes = plt.subplots(3, len(keys), figsize=(5 * len(keys), 7), sharex="col")
for c, key in enumerate(keys):
    d = detail[key]; t = np.arange(len(d["pe"])) / 10
    ax = axes[0, c]
    ax.plot(t, d["q"][:, 1], color=COLOR[key[0]], lw=1.8, label="어깨 들기 명령")
    ax.plot(t, d["meas"][:, 1], color=MUTED, lw=1.2, ls="--", label="어깨 들기 측정")
    ax.plot(t, d["q"][:, 2], color="#4a3aa7", lw=1.8, label="팔꿈치 명령")
    ax.plot(t, d["meas"][:, 2], color=MUTED, lw=1.2, ls=":", label="팔꿈치 측정")
    ax.set_title(f"{LABEL[key[0]]} ep{key[1]} ({A[key[0]][key[1]].plan.get('color','')} 바구니)", fontsize=10, color=INK2, loc="left")
    ax.set_ylabel("관절각 (°)")
    if c == 0: ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[1, c].plot(t, d["pe"], color=COLOR[key[0]], lw=1.5); axes[1, c].set_ylabel("역기구학 위치 오차 (mm)")
    axes[2, c].plot(t, d["reach"] * 100, color=COLOR[key[0]], lw=1.5); axes[2, c].set_ylabel("수평 도달 거리 (cm)")
    axes[2, c].axhline(np.percentile(reach_train, 99) * 100, color=MUTED, ls="--", lw=1)
    axes[2, c].set_xlabel("시간 (s)")
fig.tight_layout(); fig.savefig("figs/rel_ee_oscillation.png", dpi=150)
