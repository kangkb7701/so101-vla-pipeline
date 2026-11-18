"""
Test #2: 학습 데이터 첫 frame state 분포 vs 현재 home pose 비교
============================================================
각 episode의 첫 frame `observation.state` (6-dim degrees)을 추출해서
- mean / std / min / max / p01 / p99 통계
- 현재 home pose와 비교
- 각 joint별 deviation을 σ 단위로 측정 → OOD 정도 정량화

목적: home pose가 학습 분포의 어디에 위치하는지 확정.
"""
import os
import glob
import pandas as pd
import numpy as np

# ===================================================================
PARQUET_DIR = "/home/aivlab/kkb_capstone/datasets/pickup/data/chunk-000"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
HOME_POSE = np.array([24.31, -103.43, 96.13, 80.35, -105.10, 10.49])
# ===================================================================

def main():
    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "file-*.parquet")))
    print(f"Found {len(parquet_files)} parquet files\n")

    first_frame_states = []  # 각 episode의 첫 frame state
    episode_lengths = []

    for pf in parquet_files:
        df = pd.read_parquet(pf).sort_values("index").reset_index(drop=True)
        for ep_idx, ep_data in df.groupby("episode_index"):
            ep_data = ep_data.sort_values("frame_index").reset_index(drop=True)
            first_state = np.array(ep_data.iloc[0]["observation.state"], dtype=np.float64)
            first_frame_states.append(first_state)
            episode_lengths.append(len(ep_data))

    states = np.stack(first_frame_states)  # (N_episodes, 6)
    print(f"Total episodes: {len(states)}")
    print(f"Episode length: mean={np.mean(episode_lengths):.1f}, "
          f"min={min(episode_lengths)}, max={max(episode_lengths)}\n")

    print("=" * 80)
    print(f"{'Joint':<15} {'HOME':>8} {'mean':>8} {'std':>7} "
          f"{'min':>8} {'p01':>8} {'p99':>8} {'max':>8}  σ-dev  OOD?")
    print("=" * 80)

    for i, name in enumerate(JOINT_NAMES):
        col = states[:, i]
        mean = col.mean()
        std = col.std()
        mn, mx = col.min(), col.max()
        p01, p99 = np.percentile(col, [1, 99])
        sigma_dev = (HOME_POSE[i] - mean) / (std + 1e-9)
        ood_flag = ""
        if HOME_POSE[i] < p01:
            ood_flag = "BELOW_p01"
        elif HOME_POSE[i] > p99:
            ood_flag = "ABOVE_p99"
        elif HOME_POSE[i] < mn or HOME_POSE[i] > mx:
            ood_flag = "OUT_OF_RANGE"
        else:
            ood_flag = "OK"
        print(f"{name:<15} {HOME_POSE[i]:>8.2f} {mean:>8.2f} {std:>7.2f} "
              f"{mn:>8.2f} {p01:>8.2f} {p99:>8.2f} {mx:>8.2f}  "
              f"{sigma_dev:+.2f}  {ood_flag}")

    print("\n" + "=" * 80)
    print("HOME POSE vs 학습 첫 frame 분포 해석")
    print("=" * 80)
    overall_max_dev = 0
    for i, name in enumerate(JOINT_NAMES):
        col = states[:, i]
        sigma_dev = abs((HOME_POSE[i] - col.mean()) / (col.std() + 1e-9))
        if sigma_dev > overall_max_dev:
            overall_max_dev = sigma_dev
        if sigma_dev > 2.0:
            print(f"  ⚠️ {name}: home은 학습 첫 frame 평균에서 {sigma_dev:.1f}σ 떨어짐 "
                  f"(home={HOME_POSE[i]:.1f}, 학습 mean={col.mean():.1f}±{col.std():.1f})")

    print(f"\nMax σ-deviation across joints: {overall_max_dev:.2f}σ")
    if overall_max_dev > 3.0:
        print("→ OOD 강한 의심. home pose를 학습 평균에 맞추는 것 권장.")
    elif overall_max_dev > 2.0:
        print("→ 약한 OOD. perception 가설 우선 검증.")
    else:
        print("→ HOME pose는 학습 분포 안. OOD 가설 약화.")


if __name__ == "__main__":
    main()
