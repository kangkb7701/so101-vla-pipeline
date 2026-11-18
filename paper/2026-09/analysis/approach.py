"""Approach phase.

(1) Own real runs: horizontal distance of the commanded gripper position from the banana (pixel->XY map from
    grasp_analysis.py) over time, per representation.
(2) Same real inputs for all policies along the successful absolute-joint trajectories, at every 5th frame from
    the start to the grasp: predicted position 1.5 s ahead (chunk end) vs where that successful run actually was
    1.5 s later. Binned by time before the grasp.
"""
import sys, copy, json
import numpy as np, pandas as pd, torch, cv2
sys.path.insert(0, "/home/aivlab/kkb_capstone/2026-09-paper/eval")
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import bind_relative_anchor
from lerobot.policies.utils import prepare_observation_for_inference
from ee_ik_steps import install_ee_ik
from common import (PAPER, REC, EVAL, TRAIN_ROOT, REPRS, LABEL, COLOR, INK2, MUTED, fk_pos, gripper_events,
                    load_all, load_training, load_session, setup_matplotlib)

# ---- pixel -> XY map (same as grasp_analysis.py) ----
starts = {r["ep"]: r for r in json.load(open(EVAL / "train_banana_starts.json"))}
P, XY = [], []
for e, A, S in load_training():
    c, _ = gripper_events(S[:, 5], A[:, 5])
    if c and e in starts:
        P.append([starts[e]["x"], starts[e]["y"], 1.0]); XY.append(fk_pos(A[c[0]:c[0] + 1])[0, :2])
M, *_ = np.linalg.lstsq(np.array(P), np.array(XY), rcond=None)
ga = pd.read_csv("grasp_analysis.csv")

# ---- (1) own runs ----
plt = setup_matplotlib()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
curves = {}
for r, eps in load_all().items():
    D = []
    for e in eps:
        g = ga[(ga.repr == r) & (ga.ep == e.ep)].iloc[0]
        if not isinstance(g.banana_px, str):
            continue
        bx, by = map(float, g.banana_px.split(","))
        tgt = np.array([bx, by, 1.0]) @ M
        d = np.linalg.norm(fk_pos(e.action)[:, :2] - tgt, axis=1) * 100
        D.append(d)
    T = max(len(d) for d in D)
    grid = np.full((len(D), T), np.nan)
    for i, d in enumerate(D):
        grid[i, :len(d)] = d
    med = np.nanmedian(grid, 0)
    curves[r] = med
    t = np.arange(T) / 10
    ax[0].plot(t[:400], med[:400], color=COLOR[r], label=LABEL[r])
    mins = np.nanmin(grid, 1)
    print(f"{r:9s} closest horizontal approach to banana (cm): median {np.median(mins):.1f}  per-episode {np.round(mins,1).tolist()}")
ax[0].set_xlabel("시간 (s)"); ax[0].set_ylabel("그리퍼-바나나 수평 거리 (cm, 중앙값)")
ax[0].set_title("실제 시연: 그리퍼가 바나나에 다가가는 모습", fontsize=10, color=INK2, loc="left"); ax[0].legend(frameon=False, fontsize=8)

# ---- (2) same real inputs, whole approach ----
vis = pd.read_csv("outcomes_visual.csv")
src = [e for e in load_session("abs_joint", "eval") if vis[(vis.repr == "abs_joint") & (vis.ep == e.ep)].outcome.iloc[0] == "success"]
real = LeRobotDataset("local/rollout_smolvla_abs_joint_eval", root=REC / "smolvla_abs_joint" / "eval")
frm = real.meta.episodes["dataset_from_index"]
cases = []
for e in src:
    c, _ = gripper_events(e.state[:, 5], e.action[:, 5])
    pos = fk_pos(e.action)
    for t in range(0, c[0] - 15, 5):
        cases.append((e.ep, t, c[0], pos[t + 15], int(frm[e.ep]) + t))
ROOTS = {"abs_joint": TRAIN_ROOT, "rel_joint": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_rel_joint"),
         "abs_ee": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_abs_ee"), "rel_ee": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_rel_ee")}
rows = []
for r in REPRS:
    ckpt = PAPER / "runs" / f"smolvla_{r}" / "checkpoints" / "012893" / "pretrained_model"
    m = LeRobotDataset(f"local/{ROOTS[r].name}", root=ROOTS[r]).meta
    meta2 = copy.deepcopy(m)
    meta2.info["features"] = {{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}.get(k, k): v for k, v in m.features.items()}
    cfg = PreTrainedConfig.from_pretrained(ckpt); cfg.pretrained_path = str(ckpt); cfg.device = "cuda"
    policy = make_policy(cfg, ds_meta=meta2)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": "cuda"}})
    bind_relative_anchor(policy, pre); install_ee_ik(pre, post, policy.config, log_path=None)
    torch.manual_seed(0)
    for ep, t, tc, fut, idx in cases:
        it = real[idx]
        obs = {"observation.state": it["observation.state"].numpy().astype(np.float32)}
        for k in ("observation.images.camera1", "observation.images.camera2"):
            obs[k] = (it[k].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        policy.reset(); pre.reset(); post.reset()
        with torch.inference_mode():
            o = pre(prepare_observation_for_inference(obs, torch.device("cuda"), it["task"], "so_follower"))
            chunk = policy.predict_action_chunk(o)
            acts = np.array([post(chunk[:, k]).reshape(-1).cpu().numpy()[:6] for k in range(chunk.shape[1])], dtype=np.float64)
        rows.append({"repr": r, "ep": ep, "t_before_grasp_s": (tc - t) / 10,
                     "future_err_cm": float(np.linalg.norm(fk_pos(acts[-1:])[0] - fut)) * 100})
    print(r, "done", flush=True)
    del policy; torch.cuda.empty_cache()
df = pd.DataFrame(rows)
df.to_csv("approach_crosseval.csv", index=False)
bins = [0, 3, 6, 9, 30]
tab = df.groupby(["repr", pd.cut(df.t_before_grasp_s, bins)], observed=True).future_err_cm.median().unstack().round(1).reindex(REPRS)
print(tab.to_string())
for r in REPRS:
    d = df[df.repr == r].groupby(pd.cut(df[df.repr == r].t_before_grasp_s, bins), observed=True).future_err_cm.median()
    ax[1].plot([1.5, 4.5, 7.5, 12], d.values, "o-", color=COLOR[r], label=LABEL[r], ms=5)
ax[1].invert_xaxis(); ax[1].set_xlabel("잡기까지 남은 시간 (s)"); ax[1].set_ylabel("1.5초 뒤 위치 예측 오차 (cm, 중앙값)")
ax[1].set_title("같은 실제 영상 입력: 성공 궤적 대비 예측 위치 차이", fontsize=10, color=INK2, loc="left")
fig.tight_layout(); fig.savefig("figs/approach.png", dpi=150)
