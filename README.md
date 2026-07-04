# SO-101 VLA Pipeline

SO-101(SO-ARM101) 실물 로봇 VLA 추론 파이프라인. 하드웨어·카메라·IK 스택을 공유하고 policy 백엔드(Octo / ACT / VP-VLA)를 교체해서 씀.

- 로봇: `so_follower`, 6-DoF
- 데이터셋: [kangkb7701/so101-vla-datasets](https://huggingface.co/datasets/kangkb7701/so101-vla-datasets)

## 두 파이프라인

| | Octo (VLA) | ACT |
|---|---|---|
| 진입점 | `runtime/main_real2.py` + `servers/octo_server.py` | `runtime/main_act.py` |
| 프로세스 | 3개 (클라이언트 ↔ gRPC 모델서버 ↔ zmq 하드웨어서버) | 2개 (클라이언트 ↔ 하드웨어서버) |
| 모델 | `octo` + 체크포인트 | LeRobot `ACTPolicy` |
| 액션 | EE delta chunk → IK | joint 직접 |
| VP 증강 | SAM3 오버레이 + Qwen 트리거 | 없음 |

**Octo 흐름**: 카메라 → `main_real2`(VP 오버레이·이벤트 트리거 옵션) → gRPC → `octo_server`(action chunk) → `ik_ctrl`(joint 변환) → zmq → `hardware_server` → 로봇

**ACT 흐름**: 카메라 → `main_act`(ACTPolicy) → joint → zmq → `hardware_server` → 로봇

## 구조

```
so101_pipeline/
├── runtime/       진입점 — main_real2(Octo/VP-VLA), main_act(ACT), main_act_goal_onehot
├── servers/       octo_server(gRPC 모델), hardware_server(zmq 로봇)
├── agents/        정책 클라이언트 — remote_agent(Octo), vp_vla_remote_agent, qwen 트리거
├── controllers/   ik_ctrl (EE delta → joint IK)
├── envs/          real_env_client (zmq), gripper_telemetry
├── perception/    camera_source (듀얼캠), vp_runtime_overlay (SAM3 오버레이)
├── interfaces/    command_bridge (유저 명령), app_video_process (웹 비디오)
├── proto/         gRPC 계약 (vla.proto + 생성물)
└── assets/        so101_new_calib.urdf
```

루트: `configs/`, `scripts/train/`(파인튜닝), `external/VP-VLA/`(submodule), `docs/`

## 설치

```bash
git clone --recurse-submodules https://github.com/kangkb7701/so101-vla-pipeline
cd so101-vla-pipeline
pip install -e ".[act,vp]"
```

백엔드별 추가 의존성:
- **Octo**: pip 패키지 아님 → `git clone https://github.com/octo-models/octo && pip install -e octo`. 체크포인트는 repo에 없음, `checkpoints/` 아래 배치하거나 `OCTO_CHECKPOINT`로 지정
- **ACT / LeRobot**: `pip install lerobot`, 소스 쓰면 `export LEROBOT_SRC=/path/to/lerobot/src`
- **VP-VLA**: submodule `external/VP-VLA` ([JIA-Lab-research/VP-VLA](https://github.com/JIA-Lab-research/VP-VLA)) 자체 설치 안내 따름

gRPC stub 재생성 시:
```bash
python -m grpc_tools.protoc -Iso101_pipeline/proto \
  --python_out=so101_pipeline/proto --grpc_python_out=so101_pipeline/proto \
  so101_pipeline/proto/vla.proto
```

## 실행

로봇 측에서 하드웨어 서버부터 띄움.

```bash
# 1) 하드웨어 서버 (로봇 측)
python -m so101_pipeline.servers.hardware_server

# 2-A) Octo
python -m so101_pipeline.servers.octo_server
POLICY_BACKEND=octo python -m so101_pipeline.runtime.main_real2
#   VP 오버레이: USE_VP_VISUAL_PROMPT=1 (SAM3 서버 필요)

# 2-B) ACT
python -m so101_pipeline.runtime.main_act --policy-path <ACT 체크포인트>
```

설치했으면 CLI 별칭도 됨: `hardware-server`, `octo-server`, `so101-octo`, `so101-act`.

주요 환경변수: `POLICY_BACKEND`(octo/vp_vla), `USE_VP_VISUAL_PROMPT`, `ROBOT_URDF`, `OCTO_CHECKPOINT`, `LEROBOT_SRC`. 상세는 `docs/ARCHITECTURE.md` 참고.

## 학습 (재현)

`scripts/train/` — Octo 파인튜닝(`full_finetuning_so101_fruit_raw_h4_binary.py`, `finetune_so101.py`, `octo_grounding_head.py`), ACT(`train_act_so101.py`), goal one-hot 변환(`build_act_goal_onehot_dataset.py`). 데이터는 위 HF 데이터셋에서 받음.

## License

Apache-2.0 (`external/VP-VLA`는 원저작자 라이선스 따름)
