"""Same real-robot observations, four policies: where does each one aim?

Inputs: frames recorded during the successful absolute-joint evaluation episodes (real cameras, real mat, real
joint state). At every frame in the 1.5 s before that episode's grasp, each checkpoint predicts a fresh chunk
(queues reset -> the first predicted action is the chunk start; the 15th is 1.5 s ahead). The predicted gripper
position at the chunk end (FK after IK for EE) is compared with where the absolute-joint run actually grasped
the banana (its commanded gripper position at the close event, which led to a successful pick).
Offline counterpart (training frames) is in offline_replay.py; real-vs-offline gap = sensitivity to the real scene.
"""
import sys, copy
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/aivlab/kkb_capstone/2026-09-paper/eval")
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import bind_relative_anchor
from ee_ik_steps import install_ee_ik
from lerobot.policies.utils import prepare_observation_for_inference
from common import PAPER, REC, TRAIN_ROOT, REPRS, fk_pos, gripper_events, load_session

vis = pd.read_csv("outcomes_visual.csv")
src = [e for e in load_session("abs_joint", "eval") if vis[(vis.repr == "abs_joint") & (vis.ep == e.ep)].outcome.iloc[0] == "success"]
real = LeRobotDataset("local/rollout_smolvla_abs_joint_eval", root=REC / "smolvla_abs_joint" / "eval")
frm = real.meta.episodes["dataset_from_index"]
cases = []
e_grip = {}
for e in src:
    closes, _ = gripper_events(e.state[:, 5], e.action[:, 5])
    tc = closes[0]
    for t in range(max(0, tc - 15), tc - 1, 3):
        e_grip[(e.ep, t)] = e.action[t:t + 15, 5].max()
    grasp_xy = fk_pos(e.action[tc:tc + 1])[0, :2]
    for t in range(max(0, tc - 15), tc - 1, 3):   # frames 1.5 s .. 0.2 s before the grasp
        cases.append((e.ep, t, tc, grasp_xy, int(frm[e.ep]) + t))
print("episodes", len(src), "cases", len(cases))
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
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(ckpt),
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    bind_relative_anchor(policy, pre)
    install_ee_ik(pre, post, policy.config, log_path=None)
    torch.manual_seed(0)
    for ep, t, tc, gxy, idx in cases:
        it = real[idx]
        obs = {"observation.state": it["observation.state"].numpy().astype(np.float32)}
        for k in ("observation.images.camera1", "observation.images.camera2"):
            obs[k] = (it[k].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        policy.reset(); pre.reset(); post.reset()
        with torch.inference_mode():
            o = prepare_observation_for_inference(obs, torch.device("cuda"), it["task"], "so_follower")
            o = pre(o)
            chunk = policy.predict_action_chunk(o)            # (1, 15, D) normalized
            acts = []
            for k in range(chunk.shape[1]):
                acts.append(post(chunk[:, k]).reshape(-1).cpu().numpy()[:6])
        acts = np.array(acts, dtype=np.float64)
        p_end = fk_pos(acts[-1:])[0]
        rows.append({"repr": r, "ep": ep, "t_before_grasp_s": (tc - t) / 10,
                     "aim_err_cm": float(np.linalg.norm(p_end[:2] - gxy)) * 100,
                     "chunk_travel_cm": float(np.linalg.norm(fk_pos(acts[-1:])[0] - fk_pos(acts[:1])[0])) * 100,
                     "grip_max_pred": float(acts[:, 5].max()), "grip_real": float(e_grip[(ep, t)])})
    print(r, "done", flush=True)
    del policy; torch.cuda.empty_cache()
df = pd.DataFrame(rows)
df.to_csv("real_input_crosseval.csv", index=False)
print(df.groupby("repr")[["aim_err_cm", "chunk_travel_cm", "grip_max_pred", "grip_real"]].median().round(1).reindex(REPRS).to_string())
print(df.groupby(["repr", pd.cut(df.t_before_grasp_s, [0, 0.6, 1.0, 1.6])]).aim_err_cm.median().unstack().round(1).reindex(REPRS).to_string())
