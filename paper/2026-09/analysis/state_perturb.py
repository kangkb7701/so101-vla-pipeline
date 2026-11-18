"""Sensitivity to the joint-state input, on training frames (images untouched).

Same frames as camera_perturb.py. The measured joint state fed to the policy is perturbed: all arm joints by a
fixed random offset of 2 / 5 deg (same draw for every policy), shoulder_lift only by +5 deg, and the gripper
state by -10 (as if it had opened less). Reported: chunk-end gripper-position change (cm) and change of the
predicted maximum gripper opening, relative to the clean state. If a representation amplifies state errors,
small tracking errors compound in closed loop.
"""
import sys, copy
import numpy as np, pandas as pd, torch, cv2
sys.path.insert(0, "/home/aivlab/kkb_capstone/2026-09-paper/eval")
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import bind_relative_anchor
from lerobot.policies.utils import prepare_observation_for_inference
from ee_ik_steps import install_ee_ik
from common import PAPER, TRAIN_ROOT, REPRS, fk_pos, gripper_events

RENAME = {"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}
rng = np.random.default_rng(0)
U = rng.choice([-1.0, 1.0], size=5)
CONDS = {"clean": np.zeros(6), "arm2deg": np.r_[2 * U, 0], "arm5deg": np.r_[5 * U, 0], "lift+5": np.r_[0, 5, 0, 0, 0, 0],
         "grip-10": np.r_[0, 0, 0, 0, 0, -10]}


def warp(img, dx, scale):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale); M[0, 2] += dx
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


ds = LeRobotDataset("local/so101_pick_place_fruit_180", root=TRAIN_ROOT)
frm = ds.meta.episodes["dataset_from_index"]
cases = []
for ep in range(0, 180, 18):
    g = np.stack([ds.hf_dataset[int(frm[ep]) + i]["action"] for i in range(0, 400)])  # enough frames for the approach
    s = np.stack([ds.hf_dataset[int(frm[ep]) + i]["observation.state"] for i in range(0, 400)])
    c, _ = gripper_events(s[:, 5], g[:, 5])
    for t in range(max(0, c[0] - 30), c[0] - 2, 5):
        cases.append((ep, int(frm[ep]) + t))
print("cases", len(cases))
imgs = []
for ep, idx in cases:
    it = ds[idx]
    imgs.append((it["observation.state"].numpy().astype(np.float32), it["task"],
                 {k_rb: (it[k_ds].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for k_ds, k_rb in RENAME.items()}))
ROOTS = {"abs_joint": TRAIN_ROOT, "rel_joint": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_rel_joint"),
         "abs_ee": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_abs_ee"), "rel_ee": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_rel_ee")}
rows = []
for r in REPRS:
    ckpt = PAPER / "runs" / f"smolvla_{r}" / "checkpoints" / "012893" / "pretrained_model"
    m = LeRobotDataset(f"local/{ROOTS[r].name}", root=ROOTS[r]).meta
    meta2 = copy.deepcopy(m); meta2.info["features"] = {RENAME.get(k, k): v for k, v in m.features.items()}
    cfg = PreTrainedConfig.from_pretrained(ckpt); cfg.pretrained_path = str(ckpt); cfg.device = "cuda"
    policy = make_policy(cfg, ds_meta=meta2)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": "cuda"}})
    bind_relative_anchor(policy, pre); install_ee_ik(pre, post, policy.config, log_path=None)
    for (ep, idx), (state, task, im) in zip(cases, imgs):
        out = {}
        for name, d in CONDS.items():
            obs = {"observation.state": (state + d).astype(np.float32), **im}
            torch.manual_seed(0)  # same flow-matching noise for every condition
            policy.reset(); pre.reset(); post.reset()
            with torch.inference_mode():
                o = pre(prepare_observation_for_inference(obs, torch.device("cuda"), task, "so_follower"))
                chunk = policy.predict_action_chunk(o)
                acts = np.array([post(chunk[:, k]).reshape(-1).cpu().numpy()[:6] for k in range(chunk.shape[1])], dtype=np.float64)
            out[name] = (fk_pos(acts[-1:])[0], acts[:, 5].max())
        for name in CONDS:
            rows.append({"repr": r, "ep": ep, "idx": idx, "cond": name,
                         "shift_cm": float(np.linalg.norm(out[name][0] - out["clean"][0])) * 100,
                         "grip_max": out[name][1], "grip_change": out[name][1] - out["clean"][1]})
    print(r, "done", flush=True)
    del policy; torch.cuda.empty_cache()
df = pd.DataFrame(rows); df.to_csv("state_perturb.csv", index=False)
pd.set_option("display.width", 200)
print(df.groupby(["repr", "cond"]).shift_cm.median().unstack()[list(CONDS)].reindex(REPRS).round(2).to_string())
print(df.groupby(["repr", "cond"]).grip_change.median().unstack()[list(CONDS)].reindex(REPRS).round(1).to_string())
