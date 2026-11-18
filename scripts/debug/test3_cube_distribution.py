"""
Test #3: 학습 데이터 cube 위치 분포 추출 + 현재 큐브 위치 비교
============================================================
각 episode의 grasp 직전 frame (= cube 위 / 잡기 직전)에서 EE pose를 FK로 계산.
50+ episode의 EE 위치 분포 = 학습 시 cube가 놓인 위치들의 추정치.

목적: "정면 약간 떨어진 곳", "좌측 45°", "우측 45°" cube 위치가
각각 학습 분포 안인지 OOD인지 판단.

근거: pickup task의 마지막 ~10 frame에서 gripper는 cube 잡고 들어올림.
      grasp 직전 frame의 EE pose ≈ cube 위치.
"""
import os
import glob
import pandas as pd
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

# ===================================================================
PARQUET_DIR = "/home/aivlab/kkb_capstone/datasets/pickup/data/chunk-000"
URDF_PATH = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
EE_LINK = "gripper_link"

# grasp 직전을 마지막에서 N번째 frame로 가정 (cube 잡고 들어올리기 전)
GRASP_OFFSET_FROM_END = 30  # 보수적: 끝에서 30 frame 전 (= 약 1초 전)
# ===================================================================

class FK:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q_idx = []
        for name in JOINT_NAMES:
            joint_id = self.model.getJointId(name)
            self.q_idx.append(self.model.joints[joint_id].idx_q)
        self.ee_frame_id = self.model.getFrameId(EE_LINK)

    def ee_pose(self, joints_deg):
        q_rad = np.deg2rad(np.asarray(joints_deg, dtype=np.float64))
        q = np.zeros(self.model.nq)
        for i, val in enumerate(q_rad):
            q[self.q_idx[i]] = val
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.ee_frame_id]
        return pose.translation.copy()


def main():
    fk = FK(URDF_PATH)
    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "file-*.parquet")))
    print(f"Found {len(parquet_files)} parquet files")

    grasp_positions = []  # (N, 3) — EE position at grasp moment
    final_positions = []  # (N, 3) — EE position at very end (after lift)
    init_positions = []   # (N, 3) — EE position at frame 0 (home pose check)

    for pf in parquet_files:
        df = pd.read_parquet(pf).sort_values("index").reset_index(drop=True)
        for ep_idx, ep_data in df.groupby("episode_index"):
            ep_data = ep_data.sort_values("frame_index").reset_index(drop=True)
            n = len(ep_data)
            if n < GRASP_OFFSET_FROM_END + 5:
                continue
            init_state = np.array(ep_data.iloc[0]["observation.state"], dtype=np.float64)
            grasp_state = np.array(ep_data.iloc[max(0, n - GRASP_OFFSET_FROM_END)]["observation.state"], dtype=np.float64)
            final_state = np.array(ep_data.iloc[-1]["observation.state"], dtype=np.float64)

            init_positions.append(fk.ee_pose(init_state))
            grasp_positions.append(fk.ee_pose(grasp_state))
            final_positions.append(fk.ee_pose(final_state))

    init_positions = np.stack(init_positions)
    grasp_positions = np.stack(grasp_positions)
    final_positions = np.stack(final_positions)
    print(f"Total episodes processed: {len(grasp_positions)}\n")

    def stats(arr, label):
        print(f"--- {label} ---")
        for i, axis in enumerate(["x (forward)", "y (LEFT-/RIGHT+)", "z (up)"]):
            col = arr[:, i]
            print(f"  {axis:<18}: mean={col.mean():+.3f}  std={col.std():.3f}  "
                  f"min={col.min():+.3f}  p01={np.percentile(col, 1):+.3f}  "
                  f"p99={np.percentile(col, 99):+.3f}  max={col.max():+.3f}")

    stats(init_positions,  "Initial EE pose (frame 0, ~home)")
    print()
    stats(grasp_positions, f"Grasp EE pose (frame N-{GRASP_OFFSET_FROM_END}, ~cube position)")
    print()
    stats(final_positions, "Final EE pose (last frame, after lift)")

    # 좌측/우측/정면 분류
    print("\n" + "=" * 80)
    print("Cube 위치 분포 (grasp moment EE x, y 기준)")
    print("=" * 80)
    g_x = grasp_positions[:, 0]
    g_y = grasp_positions[:, 1]
    g_z = grasp_positions[:, 2]

    # 분류 (y < -0.05: LEFT, y > 0.05: RIGHT, |y| <= 0.05: CENTER)
    left_mask   = g_y < -0.05
    right_mask  = g_y >  0.05
    center_mask = np.abs(g_y) <= 0.05
    print(f"  LEFT  (y < -5cm) : {left_mask.sum():3d} episodes "
          f"({100*left_mask.mean():.0f}%)")
    print(f"  CENTER(|y| <= 5cm): {center_mask.sum():3d} episodes "
          f"({100*center_mask.mean():.0f}%)")
    print(f"  RIGHT (y > +5cm): {right_mask.sum():3d} episodes "
          f"({100*right_mask.mean():.0f}%)")

    # 거리 분포
    dist = np.sqrt(g_x**2 + g_y**2)
    print(f"\nGrasp EE 거리 (베이스 기준 xy 평면 거리): "
          f"mean={dist.mean():.3f}m, min={dist.min():.3f}m, max={dist.max():.3f}m")

    print("\n" + "=" * 80)
    print("초기 EE pose vs 현재 home pose의 EE")
    print("=" * 80)
    HOME_POSE = np.array([24.31, -103.43, 96.13, 80.35, -105.10, 10.49])
    home_ee = fk.ee_pose(HOME_POSE)
    print(f"  현재 home pose의 EE 위치: x={home_ee[0]:+.3f}, y={home_ee[1]:+.3f}, z={home_ee[2]:+.3f}")
    print(f"  학습 frame 0 EE 평균:     x={init_positions[:,0].mean():+.3f}, "
          f"y={init_positions[:,1].mean():+.3f}, z={init_positions[:,2].mean():+.3f}")
    diff = home_ee - init_positions.mean(axis=0)
    print(f"  차이: dx={diff[0]:+.3f}m, dy={diff[1]:+.3f}m, dz={diff[2]:+.3f}m")
    print(f"  → home EE 위치가 학습 시작점에서 {np.linalg.norm(diff)*100:.1f}cm 떨어져 있음")


if __name__ == "__main__":
    main()
