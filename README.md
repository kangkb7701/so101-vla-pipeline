# SO-101 VLA Pipeline

SO-101 (SO-ARM101) 실물 로봇용 **Vision-Language-Action 추론 파이프라인**.
하드웨어·카메라·IK 스택을 공유하면서 **교체 가능한 policy 백엔드**(Octo / ACT / VP-VLA)를 지원합니다.

- 🤖 로봇: `so_follower` (SO-ARM101), 6-DoF
- 📦 데이터셋: [kangkb7701/so101-vla-datasets](https://huggingface.co/datasets/kangkb7701/so101-vla-datasets)
- 🎯 두 개의 실행 파이프라인: **Octo**(복잡·VLA) / **ACT**(단순)

---

## 🧭 두 파이프라인 개요

| | **Octo (VLA)** | **ACT** |
|---|---|---|
| 진입점 | `main_real2.py` + `octo_server.py` | `main_act.py` |
| 프로세스 | 3개 (클라이언트 ↔ gRPC 모델서버 ↔ zmq 하드웨어서버) | 2개 (클라이언트 ↔ 하드웨어서버) |
| 모델 | `octo` (rail-berkeley) + 체크포인트 | LeRobot `ACTPolicy` |
| 액션 | EE delta chunk (pos3+rot3+gripper) → IK | joint action 직접 |
| VP 증강 | SAM3 오버레이 + Qwen 단계 트리거 | 없음 |
| 특징 | 정밀·범용, 셋업 복잡 | 가볍고 셋업 단순 |

### Octo 데이터 흐름
```
[top/front 카메라]
   → main_real2.py (POLICY_BACKEND=octo)
       ├─ (옵션) vp_runtime_overlay → SAM3 서버(ws) : 학습분포용 visual prompt 오버레이
       ├─ (옵션) qwen_vp_event_trigger : pick→place 단계 전환
       └─ agents/remote_agent ──gRPC──▶ octo_server.py (OctoModel, sample_actions)
             ◀── action chunk ──┘
       → controllers/ik_ctrl (EE delta → joint) ──zmq──▶ hardware_server.py → SO-101
```

### ACT 데이터 흐름
```
[카메라] → main_act.py (LeRobot ACTPolicy 로드) → joint action ──zmq──▶ hardware_server.py → SO-101
```

---

## 📁 구조

```
so101-vla-pipeline/
├── main_real2.py            # Octo/VP-VLA 런타임 진입점 (POLICY_BACKEND 환경변수)
├── main_act.py              # ACT 런타임 진입점
├── main_act_goal_onehot.py  # ACT + goal one-hot 변형 진입점
├── octo_server.py           # Octo gRPC 모델 서버
├── hardware_server.py       # 로봇 측 zmq 서버 (SO-101 팔로워 구동)
├── gripper_telemetry.py     # 그리퍼 상태 읽기
├── vp_runtime_overlay.py    # SAM3 기반 visual-prompt 오버레이 런타임
├── app_video_process.py     # (옵션) 웹 비디오 스트림 퍼블리셔
├── so101_new_calib.urdf     # 로봇 URDF (IK/캘리브)
├── proto/                   # gRPC 계약 (vla.proto + 생성물)
├── agents/                  # 정책 클라이언트: remote_agent(Octo), vp_vla_remote_agent, base, qwen 트리거
├── application/             # camera_source(듀얼캠), command_bridge(유저명령)
├── controllers/             # base_ctrl, ik_ctrl(EE→joint IK)
├── envs/                    # real_env_client(zmq 하드웨어 클라이언트)
├── configs/                 # 백엔드 설정 (VP-VLA 등)
├── scripts/train/           # Octo/ACT 파인튜닝 + 데이터셋 변환 스크립트
├── external/VP-VLA/         # (submodule) 상단 VP-VLA 백엔드 — JIA-Lab-research/VP-VLA
└── docs/ARCHITECTURE.md     # 아키텍처 상세
```

---

## ⚙️ 설치

```bash
git clone --recurse-submodules https://github.com/kangkb7701/so101-vla-pipeline
cd so101-vla-pipeline
pip install -e .            # 공통 런타임
pip install -e ".[act,vp]"  # ACT + VP 백엔드 추가

# gRPC stub 재생성이 필요하면:
python -m grpc_tools.protoc -Iproto --python_out=proto --grpc_python_out=proto proto/vla.proto
```

**백엔드별 추가 설치**

- **Octo**: pip 패키지가 아니므로 별도 설치
  ```bash
  git clone https://github.com/octo-models/octo && pip install -e octo
  ```
  체크포인트는 GitHub에 없습니다 → HF Hub/외부 스토리지에서 받아 `checkpoints/` 아래 배치.
  (`octo_server.py`가 로드하는 경로/step은 소스 상단에서 지정)
- **ACT / LeRobot**: `pip install lerobot`, 또는 소스 설치 시 `export LEROBOT_SRC=/path/to/lerobot/src`
- **VP-VLA**: submodule (`external/VP-VLA`) — [JIA-Lab-research/VP-VLA](https://github.com/JIA-Lab-research/VP-VLA), 자체 설치 안내 따름

---

## 🚀 실행

로봇 측(또는 로봇 연결 머신)에서 하드웨어 서버부터 띄웁니다.

```bash
# 1) 하드웨어 서버 (로봇 측)
python hardware_server.py

# 2-A) Octo 파이프라인
python octo_server.py                       # gRPC 모델 서버
POLICY_BACKEND=octo python main_real2.py    # 런타임 클라이언트
#   VP 오버레이 사용: USE_VP_VISUAL_PROMPT=1 (SAM3 서버 필요)

# 2-B) ACT 파이프라인
python main_act.py --policy-path <ACT 체크포인트 경로>
```

> 환경변수/플래그 상세는 각 진입점 파일 상단 및 `docs/ARCHITECTURE.md` 참고.

---

## 📦 재현 (학습)

`scripts/train/`:
- `full_finetuning_so101_fruit_raw_h4_binary.py` — 배포 Octo 체크포인트 파인튜닝
- `finetune_so101.py` — Octo 파인튜닝 템플릿
- `octo_grounding_head.py` — grounding/action head
- `train_act_so101.py` — ACT 학습
- `build_act_goal_onehot_dataset.py` — goal one-hot 데이터 변환

학습 데이터는 [so101-vla-datasets](https://huggingface.co/datasets/kangkb7701/so101-vla-datasets)에서 받습니다.

## 📄 License

Apache-2.0 (`external/VP-VLA`는 원 저작자 라이선스를 따름)
