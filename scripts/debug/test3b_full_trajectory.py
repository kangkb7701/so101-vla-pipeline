"""
Test #3b: 학습 episode의 전체 궤적 분석 + 진짜 grasp moment 식별
============================================================
이전 Test #3에서 N-30 frame이 너무 늦은 시점이었을 가능성.
이번엔:
  - 각 episode의 모든 frame EE pose 계산
  - gripper joint의 큰 변화 시점 = grasp moment 식별
  - max y reach (가장 좌우로 간 위치) 계산
  - x reach 분포

목적: 학습 시 robot이 실제로 어디까지 움직이는지 확인.
"""
import os
import glob
import pandas as pd
import numpy as np
import pinocchio as pin

PARQUET_DIR = "/home/aivlab/kkb_capstone/datasets/pickup/data/chunk-000"
URDF_PATH = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
EE_LINK = "gripper_link"


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

    grasp_positions = []  # gripper closing 직전 EE
    max_x_reaches = []    # 각 episode의 가장 멀리 간 x
    max_y_pos = []        # 가장 큰 +y (right)
    min_y_pos = []        # 가장 작은 -y (left)
    min_z_pos = []        # 가장 낮은 z (table 가까이)
    init_grippers = []
    grasp_grippers = []

    for pf in parquet_files:
        df = pd.read_parquet(pf).sort_values("index").reset_index(drop=True)
        for ep_idx, ep_data in df.groupby("episode_index"):
            ep_data = ep_data.sort_values("frame_index").reset_index(drop=True)
            n = len(ep_data)
            if n < 50:
                continue

            # 모든 frame EE pose
            states = np.stack([np.array(s, dtype=np.float64) for s in ep_data["observation.state"]])
            ee_poses = np.stack([fk.ee_pose(s) for s in states])  # (n, 3)
            grippers = states[:, 5]  # gripper joint values per frame

            # gripper 변화 추적 — 가장 빠르게 닫히는 시점 = grasp
            gripper_diff = np.diff(grippers)
            # 크게 음수 → 닫힘, 학습 데이터에서 grip이 어떻게 변하는지 알아야 하지만
            # 일단 가장 큰 |diff|를 grasp 시점으로
            grasp_idx = int(np.argmax(np.abs(gripper_diff))) + 1
            # grasp 직전 frame (cube 위 위치) = grasp_idx - 5 정도
            cube_idx = max(0, grasp_idx - 3)

            grasp_positions.append(ee_poses[cube_idx])
            max_x_reaches.append(ee_poses[:, 0].max())
            max_y_pos.append(ee_poses[:, 1].max())
            min_y_pos.append(ee_poses[:, 1].min())
            min_z_pos.append(ee_poses[:, 2].min())
            init_grippers.append(grippers[0])
            grasp_grippers.append(grippers[grasp_idx])

    grasp_positions = np.stack(grasp_positions)
    print(f"Episodes processed: {len(grasp_positions)}\n")

    print("=" * 80)
    print("진짜 grasp moment (gripper 가장 큰 변화 직전) — 추정 cube 위치")
    print("=" * 80)
    for i, axis in enumerate(["x (forward)", "y (LEFT-/RIGHT+)", "z (up)"]):
        col = grasp_positions[:, i]
        print(f"  {axis:<18}: mean={col.mean():+.3f}  std={col.std():.3f}  "
              f"min={col.min():+.3f}  p01={np.percentile(col, 1):+.3f}  "
              f"p99={np.percentile(col, 99):+.3f}  max={col.max():+.3f}")

    print("\n" + "=" * 80)
    print("Episode 전체에서 EE가 도달한 극값 (운동 범위)")
    print("=" * 80)
    print(f"  max x reach (가장 멀리)   : mean={np.mean(max_x_reaches):+.3f}  "
          f"min={min(max_x_reaches):+.3f}  max={max(max_x_reaches):+.3f}")
    print(f"  max +y reach (가장 RIGHT) : mean={np.mean(max_y_pos):+.3f}  "
          f"min={min(max_y_pos):+.3f}  max={max(max_y_pos):+.3f}")
    print(f"  max -y reach (가장 LEFT)  : mean={np.mean(min_y_pos):+.3f}  "
          f"min={min(min_y_pos):+.3f}  max={max(min_y_pos):+.3f}")
    print(f"  min z reach (table 접근)  : mean={np.mean(min_z_pos):+.3f}  "
          f"min={min(min_z_pos):+.3f}  max={max(min_z_pos):+.3f}")

    # cube 위치 분포 (실제로 어디 놓였는지)
    print("\n" + "=" * 80)
    print("Cube 위치 분포 (grasp moment 기준)")
    print("=" * 80)
    g_y = grasp_positions[:, 1]
    g_x = grasp_positions[:, 0]
    print(f"  LEFT  (y < -2cm): {(g_y < -0.02).sum():3d} eps")
    print(f"  CENTER (|y|<=2cm): {(np.abs(g_y) <= 0.02).sum():3d} eps")
    print(f"  RIGHT (y > +2cm): {(g_y > +0.02).sum():3d} eps")
    print(f"  CLOSE  (x < 0.12m): {(g_x < 0.12).sum():3d} eps")
    print(f"  MID    (0.12 <= x < 0.20): {((g_x >= 0.12) & (g_x < 0.20)).sum():3d} eps")
    print(f"  FAR    (x >= 0.20m): {(g_x >= 0.20).sum():3d} eps")

    print("\n" + "=" * 80)
    print("Gripper 변화 (grasp 동작 검증)")
    print("=" * 80)
    print(f"  Init gripper:  mean={np.mean(init_grippers):.2f}  range=[{min(init_grippers):.1f}, {max(init_grippers):.1f}]")
    print(f"  Grasp gripper: mean={np.mean(grasp_grippers):.2f}  range=[{min(grasp_grippers):.1f}, {max(grasp_grippers):.1f}]")
    print(f"  → Init→Grasp 평균 변화: {np.mean(grasp_grippers) - np.mean(init_grippers):+.2f} deg")


if __name__ == "__main__":
    main()
