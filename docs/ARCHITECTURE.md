# Architecture

## 프로세스 토폴로지

| 프로세스 | 역할 | 통신 |
|----------|------|------|
| 런타임 클라이언트 (`runtime/main_real2.py` / `main_act.py`) | 카메라 → 추론 → 액션 → 로봇 명령 | 아래 둘과 연결 |
| 모델 서버 (`servers/octo_server.py`) | Octo 체크포인트 로드, `sample_actions` | gRPC (기본 50051) |
| 하드웨어 서버 (`servers/hardware_server.py`) | SO-101 구동, 관절/그리퍼 상태 | zmq (기본 5555) |

ACT는 모델 서버 없음 — `main_act.py`가 LeRobot `ACTPolicy`를 in-process 로드.

## 컴포넌트

- **proto/** — `vla.proto`가 `Predict`/`Reset` RPC 정의. `vla_pb2*.py`는 생성물
- **agents/** — `remote_agent`(Octo gRPC 클라이언트), `vp_vla_remote_agent`(VP-VLA 백엔드), `qwen_vp_event_trigger`(Qwen2.5-VL 단계 전환)
- **controllers/ik_ctrl** — EE delta(pos3+rot3) + binary gripper → joint target (scipy IK, URDF)
- **envs/** — `real_env_client`(zmq 하드웨어 통신), `gripper_telemetry`(그리퍼 상태)
- **perception/** — `camera_source`(듀얼캠), `vp_runtime_overlay`(SAM3 텍스트 프롬프트 → 이미지 오버레이, Octo 학습분포 정합)
- **interfaces/** — `command_bridge`(유저 명령), `app_video_process`(웹/앱 MJPEG 스트림)

## 액션 스페이스

- **Octo**: EE delta chunk `(K,7)` = `[dx,dy,dz,drx,dry,drz,gripper_binary]` → `ik_ctrl`가 joint 변환
- **ACT**: joint position `[6]` 직접 출력

## 주요 환경변수

| 변수 | 의미 |
|------|------|
| `POLICY_BACKEND` | `octo` \| `vp_vla` (`main_real2`) |
| `USE_VP_VISUAL_PROMPT` | SAM3 오버레이 on/off |
| `ROBOT_URDF` | URDF 경로 (기본 `assets/so101_new_calib.urdf`) |
| `OCTO_CHECKPOINT` / `OCTO_CHECKPOINT_STEP` | Octo 체크포인트 경로/step |
| `LEROBOT_SRC` | LeRobot 소스 경로 (미설정 시 pip 패키지) |
