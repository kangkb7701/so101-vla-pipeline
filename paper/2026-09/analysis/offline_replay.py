"""Offline evaluation on training episodes (teacher-forced: observations come from the recorded demonstration).

Each representation's final checkpoint is run through the same processor pipeline as on the robot
(relative anchors and, for EE, the IK step from eval/ee_ik_steps.py), so every output is a joint command.
Compared with the demonstration's joint command at the same frame:
  - joint error per joint, overall and per phase (approach / grasp / carry / place, split at the gripper events)
  - gripper-position error (FK of predicted vs demonstrated command) in cm
  - grasp location: predicted gripper XY at its first close vs the demonstrated grasp XY
  - command jitter (lift+elbow reversals per minute) under the same inputs
This is the in-distribution reference for the real-robot numbers in grasp_analysis.py / oscillation.py.
"""
import sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/aivlab/kkb_capstone/2026-09-paper/eval")
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.common.control_utils import predict_action
from lerobot.policies.utils import make_robot_action
from lerobot.processor import bind_relative_anchor
from ee_ik_steps import install_ee_ik
from common import PAPER, TRAIN_ROOT, REPRS, JOINTS, fk_pos, gripper_events

EPISODES = list(range(0, 180, 18))  # 10 demonstrations spread over the dataset
RENAME = {"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}


def reversals(x, amp=4.0):
    n, last_ext, direction = 0, x[0], 0
    for v in x[1:]:
        d = v - last_ext
        if direction >= 0 and d < -amp: n += direction > 0; direction = -1; last_ext = v
        elif direction <= 0 and d > amp: n += direction < 0; direction = 1; last_ext = v
        elif (direction > 0 and v > last_ext) or (direction < 0 and v < last_ext): last_ext = v
    return n


def run(repr_):
    ckpt = PAPER / "runs" / f"smolvla_{repr_}" / "checkpoints" / "012893" / "pretrained_model"
    ds = LeRobotDataset("local/so101_pick_place_fruit_180", root=TRAIN_ROOT)  # observations + demonstrated joints
    meta = ds.meta
    import copy
    # features (action dim) come from the representation's own training dataset; frames are identical
    rep_root = {"abs_joint": TRAIN_ROOT, "rel_joint": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_rel_joint"),
                "abs_ee": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_abs_ee"), "rel_ee": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_rel_ee")}[repr_]
    rep_meta = LeRobotDataset(f"local/{rep_root.name}", root=rep_root).meta
    meta2 = copy.deepcopy(rep_meta)
    meta2.info["features"] = {RENAME.get(k, k): v for k, v in rep_meta.features.items()}
    cfg = PreTrainedConfig.from_pretrained(ckpt); cfg.pretrained_path = str(ckpt); cfg.device = "cuda"
    policy = make_policy(cfg, ds_meta=meta2)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(ckpt),
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    bind_relative_anchor(policy, pre)
    install_ee_ik(pre, post, policy.config, log_path=None)
    frm, to = meta.episodes["dataset_from_index"], meta.episodes["dataset_to_index"]
    rows, preds_all = [], {}
    torch.manual_seed(0)
    for ep in EPISODES:
        policy.reset(); pre.reset(); post.reset()
        P, G = [], []
        for i in range(int(frm[ep]), int(to[ep])):
            it = ds[i]
            obs = {"observation.state": it["observation.state"].numpy().astype(np.float32)}
            for k_ds, k_rb in RENAME.items():
                obs[k_rb] = (it[k_ds].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            act = predict_action(obs, policy, torch.device("cuda"), pre, post, policy.config.use_amp,
                                 task=it["task"], robot_type="so_follower")
            a = act.reshape(-1).cpu().numpy()[:6]
            P.append(a); G.append(it["action"].numpy()[:6])
        P, G = np.array(P, dtype=np.float64), np.array(G, dtype=np.float64)
        preds_all[ep] = (P, G)
        err = np.abs(P - G)
        gc, go = gripper_events(G[:, 5], G[:, 5])
        t_close = gc[0] if gc else len(G) // 2
        t_open = go[0] if go else len(G) - 1
        phases = {"approach": slice(0, max(1, t_close - 10)), "grasp": slice(max(0, t_close - 10), t_close + 10),
                  "carry": slice(t_close + 10, max(t_close + 11, t_open - 5)), "place": slice(max(0, t_open - 5), len(G))}
        fp, fg = fk_pos(P), fk_pos(G)
        pc, _ = gripper_events(P[:, 5], P[:, 5])
        row = {"repr": repr_, "ep": ep, "joint_mae": err[:, :5].mean(), "grip_mae": err[:, 5].mean(),
               "ee_err_cm": np.linalg.norm(fp - fg, axis=1).mean() * 100,
               "grasp_xy_err_cm": np.linalg.norm(fp[pc[0], :2] - fg[t_close, :2]) * 100 if pc and gc else np.nan,
               "pred_close_delay_s": (pc[0] - t_close) / 10 if pc and gc else np.nan,
               "jitter_per_min": (reversals(P[:, 1]) + reversals(P[:, 2])) / (len(P) / 600),
               "demo_jitter_per_min": (reversals(G[:, 1]) + reversals(G[:, 2])) / (len(G) / 600)}
        for name, sl in phases.items():
            row[f"mae_{name}"] = err[sl, :5].mean() if err[sl].size else np.nan
        rows.append(row)
        print(repr_, ep, {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items() if k not in ("repr", "ep")}, flush=True)
    np.savez(f"offline_preds_{repr_}.npz", **{f"ep{k}_pred": v[0] for k, v in preds_all.items()}, **{f"ep{k}_gt": v[1] for k, v in preds_all.items()})
    del policy
    torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    reprs = sys.argv[1:] or REPRS
    all_rows = []
    for r in reprs:
        t0 = time.time()
        all_rows += run(r)
        print(r, "done in", round(time.time() - t0), "s", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv("offline_replay.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.groupby("repr").mean(numeric_only=True).drop(columns="ep").round(2).to_string())
