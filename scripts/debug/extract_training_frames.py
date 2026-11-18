"""
학습 데이터 frame 추출 → debug_frames와 시각 비교용
=====================================================
각 episode에서:
  - 첫 frame (start position, robot at home, cube already placed)
  - grasp moment frame (gripper closing 직전)

용도: 학습 시 카메라가 본 장면을 보고, 추론 시 debug_frames와 비교.

출력: /home/aivlab/kkb_capstone/debug_frames/training_ref/
  ep{idx}_frame00_top.png       ← 학습 첫 frame, top camera (= image_primary)
  ep{idx}_frame00_front.png     ← 학습 첫 frame, front=wrist camera (= image_wrist)
  ep{idx}_grasp_top.png         ← grasp moment, top
  ep{idx}_grasp_front.png       ← grasp moment, front=wrist
  ep{idx}_cube_y.txt            ← 해당 episode의 추정 cube y 위치 (위치 분류용)
"""
import os
import glob
import subprocess
import numpy as np
import pandas as pd
import pinocchio as pin

DATASET_DIR = "/home/aivlab/kkb_capstone/datasets/pickup"
URDF_PATH = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
OUTPUT_DIR = "/home/aivlab/kkb_capstone/debug_frames/training_ref"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
EE_LINK = "gripper_link"

# 특정 episode만 추출 (모든 100개 추출하면 너무 많음)
# 대신 cube 위치별로 대표 episode 몇 개씩 골라서 추출
EPISODES_PER_CATEGORY = 3  # LEFT, CENTER, RIGHT 각 3개씩 = 총 9개


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


def extract_frame_ffmpeg(video_path, frame_number, output_path):
    """ffmpeg로 특정 frame 추출 → PNG 저장."""
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"select=eq(n\\,{frame_number})",
        "-vframes", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ⚠️ ffmpeg fail for {video_path} frame {frame_number}: "
              f"{result.stderr.decode()[:100]}")
        return False
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fk = FK(URDF_PATH)

    # 1. 모든 parquet 읽어서 episode 메타데이터 모으기
    parquet_files = sorted(glob.glob(os.path.join(DATASET_DIR, "data/chunk-000/file-*.parquet")))
    episodes = []  # list of (episode_idx, file_idx, parquet_df, ep_data)

    for pf in parquet_files:
        file_idx = int(os.path.basename(pf).replace("file-", "").replace(".parquet", ""))
        df = pd.read_parquet(pf).sort_values("index").reset_index(drop=True)
        df["video_frame_number"] = range(len(df))  # ffmpeg 추출 시 frame 번호
        for ep_idx, ep_data in df.groupby("episode_index"):
            ep_data = ep_data.sort_values("frame_index").reset_index(drop=True)
            # video_frame_number는 위에서 grouping 전에 설정했으므로 ep_data에 그대로 있음
            episodes.append((int(ep_idx), file_idx, ep_data))

    print(f"Total episodes: {len(episodes)}")

    # 2. 각 episode의 grasp position 계산 (cube y 분류)
    ep_meta = []
    for ep_idx, file_idx, ep_data in episodes:
        n = len(ep_data)
        if n < 50:
            continue
        states = np.stack([np.array(s, dtype=np.float64)
                           for s in ep_data["observation.state"]])
        grippers = states[:, 5]
        gripper_diff = np.diff(grippers)
        grasp_idx = int(np.argmax(np.abs(gripper_diff))) + 1
        cube_idx = max(0, grasp_idx - 3)
        cube_pos = fk.ee_pose(states[cube_idx])
        # video_frame_number는 parquet 전체에서의 frame index (ffmpeg input)
        first_vfn = int(ep_data.iloc[0]["video_frame_number"])
        grasp_vfn = int(ep_data.iloc[cube_idx]["video_frame_number"])
        ep_meta.append({
            "ep_idx": ep_idx,
            "file_idx": file_idx,
            "first_vfn": first_vfn,
            "grasp_vfn": grasp_vfn,
            "cube_y": float(cube_pos[1]),
            "cube_x": float(cube_pos[0]),
            "cube_z": float(cube_pos[2]),
            "n_frames": n,
        })

    # 3. cube y 위치별로 분류 + 대표 N개씩 선택
    left_eps   = sorted([e for e in ep_meta if e["cube_y"] < -0.05], key=lambda e: e["cube_y"])
    center_eps = sorted([e for e in ep_meta if abs(e["cube_y"]) <= 0.05], key=lambda e: abs(e["cube_y"]))
    right_eps  = sorted([e for e in ep_meta if e["cube_y"] > 0.05], key=lambda e: -e["cube_y"])

    print(f"  LEFT episodes:   {len(left_eps)} (cube_y < -5cm)")
    print(f"  CENTER episodes: {len(center_eps)} (|cube_y| <= 5cm)")
    print(f"  RIGHT episodes:  {len(right_eps)} (cube_y > +5cm)")

    selected = []
    for category, eps in [("LEFT", left_eps), ("CENTER", center_eps), ("RIGHT", right_eps)]:
        # 가장 극단인 N개
        for e in eps[:EPISODES_PER_CATEGORY]:
            e_with_cat = dict(e)
            e_with_cat["category"] = category
            selected.append(e_with_cat)

    print(f"\nSelected {len(selected)} episodes for frame extraction:")
    for e in selected:
        print(f"  {e['category']:>6} ep{e['ep_idx']:03d} file{e['file_idx']:03d}  "
              f"cube=({e['cube_x']:+.3f}, {e['cube_y']:+.3f}, {e['cube_z']:+.3f})  "
              f"first_vfn={e['first_vfn']} grasp_vfn={e['grasp_vfn']}")

    # 4. 각 episode에서 first frame + grasp frame 추출 (top + front)
    print(f"\nExtracting frames to {OUTPUT_DIR}/ ...")
    for e in selected:
        prefix = f"{e['category']}_ep{e['ep_idx']:03d}"
        # video paths
        for cam_label, cam_dir in [("top", "observation.images.top"),
                                   ("front", "observation.images.front")]:
            vid = os.path.join(DATASET_DIR, "videos", cam_dir,
                               f"chunk-000/file-{e['file_idx']:03d}.mp4")
            if not os.path.exists(vid):
                print(f"  ⚠️ video missing: {vid}")
                continue
            # 첫 frame
            out_first = os.path.join(OUTPUT_DIR, f"{prefix}_first_{cam_label}.png")
            extract_frame_ffmpeg(vid, e["first_vfn"], out_first)
            # grasp frame
            out_grasp = os.path.join(OUTPUT_DIR, f"{prefix}_grasp_{cam_label}.png")
            extract_frame_ffmpeg(vid, e["grasp_vfn"], out_grasp)

        # cube 위치 메타정보 텍스트
        meta_txt = os.path.join(OUTPUT_DIR, f"{prefix}_meta.txt")
        with open(meta_txt, "w") as f:
            f.write(f"category={e['category']}\n")
            f.write(f"episode_idx={e['ep_idx']}\n")
            f.write(f"file_idx={e['file_idx']}\n")
            f.write(f"cube_position=({e['cube_x']:+.4f}, {e['cube_y']:+.4f}, {e['cube_z']:+.4f}) m\n")
            f.write(f"first_frame_video_index={e['first_vfn']}\n")
            f.write(f"grasp_frame_video_index={e['grasp_vfn']}\n")
            f.write(f"episode_length={e['n_frames']} frames\n")

    print(f"\n✅ Done. Check {OUTPUT_DIR}/")
    print(f"   각 episode당 4개 PNG (first_top, first_front, grasp_top, grasp_front) + meta.txt")


if __name__ == "__main__":
    main()
