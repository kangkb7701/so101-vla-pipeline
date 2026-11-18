"""
Test #2b: 실제 robot 부팅 시 home pose vs 학습 첫 frame 분포 재비교
=============================================================
이전 Test #2는 genesis_env.py:87 (시뮬용) 값을 잘못 가져옴.
실제 hardware_server 부팅 시 출력된 자세로 재분석.
"""
import os, glob
import pandas as pd
import numpy as np
import pinocchio as pin

PARQUET_DIR = "/home/aivlab/kkb_capstone/datasets/pickup/data/chunk-000"
URDF_PATH = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
EE_LINK = "gripper_link"

# 실제 hardware_server 부팅 시 출력된 자세 (사용자 제공)
ACTUAL_HOME = np.array([2.42, -104.57, 97.45, 74.90, -166.20, 0.00])
# 이전에 잘못 사용한 시뮬 값 (참조용)
SIM_HOME = np.array([24.31, -103.43, 96.13, 80.35, -105.10, 10.49])


class FK:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q_idx = []
        for name in JOINT_NAMES:
            self.q_idx.append(self.model.joints[self.model.getJointId(name)].idx_q)
        self.ee_frame_id = self.model.getFrameId(EE_LINK)

    def ee_pose(self, joints_deg):
        q_rad = np.deg2rad(np.asarray(joints_deg, dtype=np.float64))
        q = np.zeros(self.model.nq)
        for i, val in enumerate(q_rad):
            q[self.q_idx[i]] = val
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id].translation.copy()


def main():
    # 학습 첫 frame state 분포 추출
    files = sorted(glob.glob(os.path.join(PARQUET_DIR, "file-*.parquet")))
    first_states = []
    for pf in files:
        df = pd.read_parquet(pf).sort_values("index").reset_index(drop=True)
        for ep_idx, ep_data in df.groupby("episode_index"):
            ep_data = ep_data.sort_values("frame_index").reset_index(drop=True)
            first_states.append(np.array(ep_data.iloc[0]["observation.state"], dtype=np.float64))
    states = np.stack(first_states)

    print(f"학습 episodes: {len(states)}")
    print()
    print("=" * 100)
    print(f"{'Joint':<14} {'ACTUAL':>9} {'SIM(잘못)':>10} {'학습mean':>10} {'std':>7} "
          f"{'min':>9} {'p01':>9} {'p99':>9} {'max':>9}  {'σ(actual)':>10}  {'OOD?':<10}")
    print("=" * 100)
    for i, name in enumerate(JOINT_NAMES):
        col = states[:, i]
        mean, std = col.mean(), col.std()
        mn, mx = col.min(), col.max()
        p01, p99 = np.percentile(col, [1, 99])
        sigma = (ACTUAL_HOME[i] - mean) / (std + 1e-9)
        if ACTUAL_HOME[i] < mn:
            ood = "BELOW_min"
        elif ACTUAL_HOME[i] > mx:
            ood = "ABOVE_max"
        elif ACTUAL_HOME[i] < p01:
            ood = "<p01"
        elif ACTUAL_HOME[i] > p99:
            ood = ">p99"
        else:
            ood = "OK"
        print(f"{name:<14} {ACTUAL_HOME[i]:>9.2f} {SIM_HOME[i]:>10.2f} "
              f"{mean:>10.2f} {std:>7.2f} {mn:>9.2f} {p01:>9.2f} {p99:>9.2f} {mx:>9.2f}  "
              f"{sigma:>+9.2f}σ  {ood:<10}")

    # EE pose 비교
    fk = FK(URDF_PATH)
    actual_ee = fk.ee_pose(ACTUAL_HOME)

    # 학습 frame 0 EE 평균/통계
    init_ees = np.stack([fk.ee_pose(s) for s in states])
    print()
    print("=" * 100)
    print("EE pose 비교 (학습 첫 frame vs 실제 robot home)")
    print("=" * 100)
    print(f"  학습 frame 0 EE: mean=({init_ees[:,0].mean():+.3f}, "
          f"{init_ees[:,1].mean():+.3f}, {init_ees[:,2].mean():+.3f})  "
          f"std=({init_ees[:,0].std():.3f}, {init_ees[:,1].std():.3f}, {init_ees[:,2].std():.3f})")
    print(f"  학습 frame 0 EE range: x=[{init_ees[:,0].min():+.3f}, {init_ees[:,0].max():+.3f}]  "
          f"y=[{init_ees[:,1].min():+.3f}, {init_ees[:,1].max():+.3f}]  "
          f"z=[{init_ees[:,2].min():+.3f}, {init_ees[:,2].max():+.3f}]")
    print(f"  실제 robot home EE: ({actual_ee[0]:+.3f}, {actual_ee[1]:+.3f}, {actual_ee[2]:+.3f})")
    diff = actual_ee - init_ees.mean(axis=0)
    print(f"  차이: dx={diff[0]:+.3f}m, dy={diff[1]:+.3f}m, dz={diff[2]:+.3f}m  "
          f"(거리={np.linalg.norm(diff)*100:.1f}cm)")

    # 실제 robot EE가 학습 EE 분포 안인지
    in_x = init_ees[:,0].min() <= actual_ee[0] <= init_ees[:,0].max()
    in_y = init_ees[:,1].min() <= actual_ee[1] <= init_ees[:,1].max()
    in_z = init_ees[:,2].min() <= actual_ee[2] <= init_ees[:,2].max()
    print(f"  분포 안 여부: x={in_x}, y={in_y}, z={in_z}")


if __name__ == "__main__":
    main()
