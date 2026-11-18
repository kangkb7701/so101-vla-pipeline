"""Did the robot follow the commanded joints? (all representations; recorded command vs measured joints)

- lag: mean |command(t) - measured(t+k)| for k = 0..5 frames (10 Hz) -> which delay fits best
- steady-state error with the object in the gripper vs without (gripper commanded closed but measured stays open
  by > 5 = something is held)
- chunk boundaries: the policy re-plans every 15 frames; the phase (t mod 15) with the largest command jump
  is taken as the boundary; boundary jump vs within-chunk step size
- for EE runs also the task-space gap: FK(command) vs FK(measured) in mm
"""
import numpy as np, pandas as pd
from common import REPRS, LABEL, COLOR, INK2, ARM, load_all, load_training, fk_pos, setup_matplotlib

A = load_all()
out, lag_rows = [], []
for r in REPRS:
    eps = A[r]
    lags = np.zeros((6, 5))
    for k in range(6):
        errs = [np.abs(e.action[:len(e.action) - k, :5] - e.state[k:, :5]) for e in eps]
        lags[k] = np.concatenate(errs).mean(0)
    best_k = int(np.argmin(lags.mean(1)))
    for k in range(6):
        lag_rows.append({"repr": r, "k_frames": k, **{j: round(v, 2) for j, v in zip(ARM, lags[k])}, "mean": round(lags[k].mean(), 2)})
    held, free = [], []
    jump_b, jump_w, phase_hist = [], [], np.zeros(15)
    for e in eps:
        k = best_k
        err = np.abs(e.action[:len(e.action) - k, :5] - e.state[k:, :5])
        hold = (e.action[:len(e.action) - k, 5] < 8) & (e.state[k:, 5] > e.action[:len(e.action) - k, 5] + 5)
        held.append(err[hold]); free.append(err[~hold])
        step = np.abs(np.diff(e.action[:, :5], axis=0)).max(1)
        for p in range(15):
            phase_hist[p] += step[p::15].mean() if len(step[p::15]) else 0
    phase = int(np.argmax(phase_hist))
    for e in eps:
        step = np.abs(np.diff(e.action[:, :5], axis=0)).max(1)
        idx = np.arange(len(step))
        jump_b.append(step[idx % 15 == phase]); jump_w.append(step[idx % 15 != phase])
    held, free = np.concatenate(held), np.concatenate(free)
    jb, jw = np.concatenate(jump_b), np.concatenate(jump_w)
    row = {"repr": r, "episodes": len(eps), "best_lag_frames": best_k,
           "err_at_best_lag_deg": round(lags[best_k].mean(), 2),
           "err_free_deg": round(free.mean(), 2), "err_holding_deg": round(held.mean(), 2) if len(held) else None,
           "holding_frames": len(held), "boundary_phase": phase,
           "boundary_jump_p50": round(np.median(jb), 2), "boundary_jump_p95": round(np.percentile(jb, 95), 2),
           "within_step_p50": round(np.median(jw), 2), "within_step_p95": round(np.percentile(jw, 95), 2),
           "phase_contrast": round(phase_hist[phase] / np.delete(phase_hist, phase).mean(), 2)}
    if r.endswith("_ee"):
        gaps = []
        for e in eps:
            gaps.append(np.linalg.norm(fk_pos(e.action[:len(e.action) - best_k]) - fk_pos(e.state[best_k:]), axis=1) * 1000)
        g = np.concatenate(gaps)
        row.update({"task_gap_mm_p50": round(np.median(g), 1), "task_gap_mm_p95": round(np.percentile(g, 95), 1)})
    out.append(row)

# training reference: teleop command vs follower measured (same definition)
tr = load_training()
lags_t = [np.concatenate([np.abs(a[:len(a) - k, :5] - s[k:, :5]) for _, a, s in tr]).mean() for k in range(6)]
print("training (teleop) command->measured error by lag k:", np.round(lags_t, 2))
df = pd.DataFrame(out)
pd.set_option("display.width", 250)
print(df.to_string(index=False))
lag = pd.DataFrame(lag_rows)
lag.to_csv("tracking_lag.csv", index=False)
df.to_csv("tracking_summary.csv", index=False)

plt = setup_matplotlib()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for r in REPRS:
    d = lag[lag.repr == r]
    ax[0].plot(d.k_frames * 100, d["mean"], "o-", color=COLOR[r], label=LABEL[r], ms=5)
ax[0].plot(np.arange(6) * 100, lags_t, "o--", color="#898781", label="학습 데이터(원격조종)", ms=4, lw=1.5)
ax[0].set_xlabel("명령 후 경과 시간 (ms)"); ax[0].set_ylabel("명령-측정 관절 오차 평균 (°)")
ax[0].set_title("명령한 관절각에 몇 ms 뒤 도달하나", fontsize=10, color=INK2, loc="left"); ax[0].legend(frameon=False, fontsize=8)
x = np.arange(4)
ax[1].bar(x - 0.18, df.within_step_p95, 0.34, color=[COLOR[r] for r in REPRS], alpha=0.45, label="청크 내부 step (p95)")
ax[1].bar(x + 0.18, df.boundary_jump_p95, 0.34, color=[COLOR[r] for r in REPRS], label="청크 경계 (p95)")
ax[1].set_xticks(x); ax[1].set_xticklabels([LABEL[r] for r in REPRS]); ax[1].set_ylabel("한 step 명령 변화 최대 관절 (°)")
ax[1].set_title("청크가 바뀌는 순간 명령이 튀는 정도", fontsize=10, color=INK2, loc="left"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figs/tracking.png", dpi=150)
