# Architecture

SO-101 VLA 파이프라인의 구성 요소와 프로세스 간 통신 구조.

## 프로세스 토폴로지

세 종류의 프로세스가 네트워크로 연결됩니다.

| 프로세스 | 역할 | 통신 |
|----------|------|------|
| **런타임 클라이언트** (`main_real2.py` / `main_act.py`) | 카메라 캡처 → 정책 추론 요청 → 액션 → 로봇 명령 | 아래 둘과 연결 |
| **모델 서버** (`octo_server.py`) | Octo 체크포인트 로드, `sample_actions` | gRPC (`proto/vla.proto`, 기본 50051) |
| **하드웨어 서버** (`hardware_server.py`) | SO-101 팔로워 구동, 관절 상태/그리퍼 텔레메트리 | zmq (기본 5555) |

ACT 백엔드는 모델 서버가 없습니다 — `main_act.py`가 LeRobot `ACTPolicy`를 in-process로 로드합니다.

## 컴포넌트

- **proto/** — `vla.proto`가 `Predict`/`Reset` RPC 계약을 정의. `vla_pb2*.py`는 생성물.
- **agents/**
  - `base_agent.py` — 정책 클라이언트 추상 베이스
  - `remote_agent.py` — Octo gRPC 클라이언트 (`octo_server`와 통신)
  - `vp_vla_remote_agent.py` — VP-VLA 백엔드 클라이언트 (`external/VP-VLA` 서버)
  - `qwen_vp_event_trigger.py` — Qwen2.5-VL 기반 pick→place 단계 전환 판단
- **controllers/** — `ik_ctrl.py`가 EE delta(pos3+rot3) + binary gripper → joint target 변환 (scipy IK, URDF 사용)
- **envs/real_env_client.py** — zmq로 하드웨어 서버에 joint 명령 전송/관측 수신
- **application/**
  - `camera_source.py` — top/front(또는 wrist) 듀얼 카메라 오픈
  - `command_bridge.py` — 외부(앱/CLI) 유저 명령 브리지
- **vp_runtime_overlay.py** — SAM3 텍스트 프롬프트로 이미지에 crosshair/box 오버레이 (Octo 학습분포와 입력 정합)
- **app_video_process.py** — (옵션) 웹/앱용 MJPEG 스트림 퍼블리셔

## 액션 스페이스

- **Octo**: end-effector delta chunk `(K, 7)` = `[dx, dy, dz, drx, dry, drz, gripper_binary]` → `ik_ctrl`가 joint로 변환
- **ACT**: LeRobot 액션 스페이스의 joint position `[6]` 직접 출력

## 주요 환경변수 / 설정

| 변수 | 의미 |
|------|------|
| `POLICY_BACKEND` | `octo` \| `vp_vla` (`main_real2.py`) |
| `USE_VP_VISUAL_PROMPT` | SAM3 visual-prompt 오버레이 on/off |
| `LEROBOT_SRC` | LeRobot 소스 경로 (미설정 시 설치된 pip 패키지 사용, `main_act.py`) |

체크포인트 경로/step 등 모델별 세부는 `octo_server.py` 상단 및 각 진입점 파일 참고.
