"""IK accuracy on ground truth: EE target = FK(recorded joint action), initial guess = recorded joint state.
Reports how well EEToJointsIKStep.solve recovers the recorded joint action. Run from 2026-09-paper/eval."""
import sys
import numpy as np
from lerobot.datasets import LeRobotDataset
from ee_ik_steps import CacheJointStateStep, EEToJointsIKStep

D = "/home/aivlab/kkb_capstone/data/lerobot"
ee = LeRobotDataset("local/so101_pick_place_fruit_180_abs_ee", root=f"{D}/so101_pick_place_fruit_180_abs_ee").hf_dataset
jt = LeRobotDataset("local/so101_pick_place_fruit_180", root=f"{D}/so101_pick_place_fruit_180").hf_dataset
idx = np.random.default_rng(0).choice(len(ee), int(sys.argv[1]) if len(sys.argv) > 1 else 2000, replace=False)
A_ee = np.array(ee.select(idx)["action"], dtype=np.float32)
S = np.array(jt.select(idx)["observation.state"], dtype=np.float32)
A_j = np.array(jt.select(idx)["action"], dtype=np.float32)
ik = EEToJointsIKStep(cache=CacheJointStateStep(), joint_limits=sys.argv[2] if len(sys.argv) > 2 else "urdf")
pos, rot, jerr, clip, fb = [], [], [], 0, 0
for a, s, g in zip(A_ee, S, A_j):
    q, info = ik.solve(a, s[:5])
    pos.append(info["pos_err_mm"]); rot.append(info["rot_err_deg"]); jerr.append(np.abs(q - g[:5]))
    clip += info["clipped"]; fb += info["fallback"]
pos, rot, jerr = np.array(pos), np.array(rot), np.array(jerr)
print(f"frames {len(idx)} | pos err mm: median {np.median(pos):.2f} p95 {np.percentile(pos,95):.2f} max {pos.max():.2f}")
print(f"rot err deg: median {np.median(rot):.2f} p95 {np.percentile(rot,95):.2f} max {rot.max():.2f}")
print("joint err vs recorded action (deg) median", np.round(np.median(jerr, 0), 2), "p95", np.round(np.percentile(jerr, 95, 0), 2))
print(f"clipped {clip} fallback {fb}")
