# archive

kkb_capstone 로컬 작업본을 PC에서 지우기 전에 보존한 변경분.

- `kkb_src_uncommitted.patch`: kkb_capstone/src(miseong2/VLARobotArm, pre-demo-snapshot 브랜치)의 커밋 안 된 수정분. ik_ctrl gripper absolute 모드, hardware_server 관절 안전 범위(2026-09-08 기준), main_real2 ACTION_SPACE=joint, octo_server 수정 포함. so101_pipeline 쪽 파일에는 아직 반영 안 됨.
- `thirdparty_*.patch`: 외부 저장소 클론(lerobot groot 수정, octo_server.py, gym-so100 kkb_scripts, SO-ARM100 urdf 등)의 로컬 수정분.
