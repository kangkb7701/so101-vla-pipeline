# 원격 구동 구현 기록 — 노트북 전환 검증부터 실팔 테스트까지

`REMOTE_DEPLOYMENT_DISCUSSION.md`의 후속. 그 문서의 계획을 검증하고, 설계를 일부 뒤집고, 실제로 구현·테스트한 기록. 대화 순서대로.

---

## 1. 젯슨→노트북 전환 판단 검토

전환 자체는 타당. 결정적 근거(노트북에는 lerobot이 그냥 깔려서 feetech 드라이버·캘리브레이션 변환을 손으로 재현할 필요가 없음)와 부수 근거(USB3 병목, 전원, aarch64 휠)의 우선순위도 맞음.

다만 원문서에 구멍 두 개:

- **"노트북에서 GPU를 쓸 수 없다"가 측정 없이 전제로 박혀 있었다.** 이 전제 하나 때문에 문서 전체(청크 모드, 워치독, 터널, 프록시)가 필요해졌는데 정작 검증이 없었다.
- **"서버 정책 스택은 손 안 댄다"는 성립하지 않는다.** 자동정지가 틱 카운트 기반(10Hz 전제)이라 추론 주파수를 바꾸면 재튜닝/로직 수정이 필요하고, 앱 영상·명령 경로가 논의에서 빠져 있었다.

## 2. ACT CPU 추론 실측

노트북 = i7-1165G7(4C/8T) + Iris Xe, NVIDIA GPU 없음. 체크포인트가 없어 동일 구조(resnet18+FrozenBN, enc4/dec1, dim512, 쿼리100)를 세워 측정. torch 2.13 CPU, 8스레드, 카메라 2대, batch 1:

| 해상도 | median | p95 | Hz |
|---|---|---|---|
| 640×480 | 292ms | 326ms | 3.4 |
| 320×240 | 97ms | 124ms | 10.4 |
| 224×224 | 85ms | 100ms | 11.8 |

- `torch.set_num_threads(8)`이 기본값 4보다 20~25% 빠름
- 640×480 비용 분해: resnet18 304ms + 트랜스포머 172ms

**결론: 로컬 CPU 실행 가능 여부는 학습 해상도 하나에 달렸다.**

## 3. 학습 해상도 확인 — 480×640 원본 그대로

HF 데이터셋(`fruit/pick-place/act-goal-onehot`, `act-extra`)의 `meta/info.json` 직접 확인: 두 카메라 모두 shape `[480, 640, 3]`, **av1**, fps 10. (원문서의 "h264 저장" 서술은 부정확하나 논지는 유효.)

ACT 경로 어디에도 리사이즈가 없음 — lerobot `ACTConfig`에 리사이즈 필드 자체가 없고, goal-onehot 변환 스크립트는 비디오를 하드링크로 그대로 복사. **낮은 해상도 기억은 다른 백엔드 것**: VP-VLA가 224×224, Octo가 256/128.

→ 640×480에서 CPU 3.4Hz이므로 로컬 추론 불가. **서버 추론 확정.** 동시에 "모델 입력 크기로 줄여 보내면 대역폭 해결"이라는 원문서의 논거도 죽음(줄일 여지가 없음).

## 4. 코덱·대역폭·재학습 판단

실측 JPEG(실제 카메라 프레임, 480×640 두 장 합계):

| 품질 | 합계 KB | 3Hz | 5Hz | 10Hz |
|---|---|---|---|---|
| 70 | 71 | 1.7M | 2.9M | 5.8M |
| 50 | 48 | 1.2M | 2.0M | 3.9M |

인코딩은 두 장 1.3ms로 무시 가능. 병목은 오직 **업링크 대역폭 × 추론 주파수**.

- 코덱의 대가는 지연이 아니라 **유실 내성**(H.264 인터프레임 의존 → 끊기면 뭉개진 프레임, 이게 유실보다 나쁨). 단 키프레임 1초 간격이면 대부분 완화되고, 이 크기에서는 H.264가 JPEG의 절반 이하 대역폭.
- **320×240 재학습은 마지막 수단으로 강등.** 검증 끝난 정책을 버리는 일이고, 3090에서 480×640 추론은 이미 10~20ms라 "서버도 빨라진다"는 근거는 틀렸음.
- 순서: 핫스팟 실측 → JPEG q60~70 → 부족하면 H.264(키프레임 1s) → 3Hz → 최후에 재학습.

## 5. 10Hz 왕복은 포기, 청크 모드로

10Hz 왕복(100ms 예산)은 핫스팟에서 불가(업링크 전송시간만 ~58ms@10Mbps + RTT). 그러나 **이 요구 자체가 잘못**: 100ms 예산은 로컬 제어 루프에만 적용되고, 네트워크는 루프 밖으로 뺀다.

- **적응형 추론(one-in-flight)**: 고정 주파수 대신 "요청 하나만 대기" — 이전 청크가 도착하면 다음 관측 전송. 링크 품질에 자동 적응, 요청 적체 없음.
- **시간 정렬**: ACT는 절대 관절각 출력이라 드리프트가 아니라 시간 정렬 문제. 관측에 t_obs(노트북 모노토닉)를 붙여 보내고 서버가 에코 → 노트북이 (now−t_obs)×fps로 인덱싱해 지연만큼 스텝을 건너뜀. 시계 동기화 불필요.
- **청크는 버퍼**: chunk_size=100(10초)이지만 소비 상한 10스텝(1초). 청크 길이는 재생 분량이 아니라 버틸 수 있는 끊김 길이. 상한 초과 시 홀드 → 링크 5초 끊기면 홈복귀.
- 잔여 오차는 스텝당 클램프 + 다음 관측이 흡수. 남는 본질적 비용은 "판단 근거가 Δ만큼 낡음"뿐 — 정적 장면(바나나)이라 관대.
- 자동정지는 틱 기반 → **시간 기반**으로 이식. chunk-stability 정렬은 t_obs 차이만큼 오프셋.

## 6. 하드웨어 검증 (노트북, 실기기)

- 시리얼 COM3(CH343) 1Mbps, 서보 ID 1~6 전원 응답, 캘리브레이션 `is_calibrated=True`
- 카메라: MSMF 백엔드에서만 전부 열림(DSHOW는 일부 실패). 2대 동시 각 30fps, 3대도 가능 → USB 대역폭 우려 기각
- 60초 동시 부하(카메라 2대 30fps + 서보 10Hz 폴링): 시리얼 타임아웃 **0회**, 6서보 폴링 median 3.0ms → 시리얼 굶주림 우려 기각. 허브 구성 그대로 사용 가능
- **카메라 인덱스는 재연결마다 셔플됨** (한 번은 0=내장/1=top/2=wrist, 다음엔 0=wrist/1=내장/2=top). 고정값 금지, 매 세션 확인 필요 → `--identify` 도구로 해결

## 7. 환경 구축 (conda env `lerobot312`)

Python 3.12.13 + lerobot 0.6.1 + `[dataset]` + `[feetech]` + torch 2.11 cpu + websockets + msgpack + 프로젝트 런타임 의존성. `main_act.py` 전체 import 통과.

발견한 함정:
- **lerobot 0.5.0부터 Python 3.12 이상 요구.** 3.11에서는 pip이 조용히 0.4.4로 다운그레이드
- `so_follower`는 0.4.4에서 도입(0.4.3에는 없음). 구버전의 `so100/so101_follower`가 통합된 것
- **lerobot이 의존하는 `opencv-python-headless`는 MSMF가 동작 안 함** → full `opencv-python`으로 교체 필수
- 0.6.1에서 `OpenCVCameraConfig.backend` 기본값이 ANY(구버전의 Windows→MSMF 자동선택 삭제) → MSMF 명시
- MJPG fourcc는 MSMF에서 안 먹힘(경고 후 무시) — 그래도 2대 30fps 나오므로 실사용 무관
- 구 conda env `lerobot`(py3.10)은 OpenSSL 3.6.3이 깨져 `import lerobot` 불가 — 폐기
- **Windows 콘솔 cp949: print에 em-dash 등 비ASCII → UnicodeEncodeError로 프로세스 사망.** `PYTHONIOENCODING=utf-8` 권장

## 8. 아키텍처 확정 — 앱은 노트북에, 서버는 무상태로

원래 계획(앱→서버 유지, 서버가 정책 루프 소유, 로봇만 프록시)을 **뒤집음**. 최종:

```
앱(폰) ←로컬 WiFi→ 노트북 ←ws(터널)→ 서버(GPU)
```

- **앱 → 노트북**: 명령(/command/*)과 영상(/video_feed) 모두. 앱은 단일 baseUrl만 쓰므로 IP 입력만 바꾸면 무수정. 정지 버튼이 WAN을 안 탐(안전상 결정적). 앱 영상(~19Mbps급)이 WAN에서 제거됨
- **노트북**: 하드웨어 소유, 10Hz 로컬 루프, 청크 재생, 안전장치 전부, 자동정지 판정, 앱 백엔드, 상태 주인
- **서버**: 무상태 RPC — 관측 in, 청크 out. episode_id 바뀔 때만 policy.reset(). 재시작 자유, 단독 테스트 가능. π 계열 policy server와 같은 패턴
- 연결은 노트북이 밖으로 거는 websocket 하나(CGNAT 대응). 서버→노트북 제어 메시지 불필요(그 결정을 노트북이 직접 하므로 프로토콜이 요청-응답 하나로 축소)

앱 백엔드 원본 코드는 랩 서버에만 있어서, Flutter 앱 코드에서 스키마를 역산해 새로 작성.

## 9. 구현 (레포에 추가된 파일)

- **`so101_pipeline/runtime/main_edge.py`** — 노트북 에이전트. 카메라 `--identify`(매핑을 edge_cameras.json에 저장), ws 정책 클라이언트(one-in-flight, 백오프 재접속), 10Hz 재생(t_obs 인덱싱), 클램프(팔 6°/틱, 그리퍼 25/틱) + 관절 절대 리미트 + 소비 상한 10스텝 + 워치독(첫 청크 5s 타임아웃 / 수신 5s 두절 시 홈), 시간 기반 자동정지, 로컬 홈복귀(2.5s@30Hz), `--dry_run`
- **`so101_pipeline/servers/act_policy_server.py`** — 서버 정책 서비스. msgpack over websocket(:8765), main_act의 정책 스택 그대로, `--mock`이면 GPU 없이 hold-position 청크
- **`so101_pipeline/interfaces/edge_app_backend.py`** — 앱 백엔드(:8000 한 포트): /health, /command/voice·stop·latest·history·success, /video_feed(MJPEG)

## 10. 검증 결과

**mock + dry-run** (실기기 읽기, 쓰기 없음): 4개 안전 경로 전부 통과
1. 정상 흐름: 명령 → 청크 age ~0.1s → 10s에 자동정지(stable chunk) → 성공 기록(앱 반영)
2. 앱 정지: 즉시 중단 → 홈
3. 링크 단절(서버 킬): 소비 상한에서 홀드 → 5s 후 안전 정지 → 홈 → 백오프 재접속
4. 첫 청크 타임아웃: 5s 미수신 시 중단

**mock + 실팔** (2026-08-19): 에피소드 완주. 토크 인가 튐 없음, 10초 유지 clamps=0/holds=0, 자동정지 후 홈복귀 부드러움.

## 11. 남은 작업

1. **랩 서버**: 레포 pull, `pip install websockets msgpack`, 실체크포인트로 정책 서버 기동
   ```bash
   python -m so101_pipeline.servers.act_policy_server \
     --policy_path <체크포인트> --dataset_repo_id <repo> --dataset_root <경로> --device cuda
   ```
2. **노트북**: `set ACT_SERVER_URL=ws://<랩서버IP>:8765`로 접속 대상만 변경
3. **랩 내 유선/WiFi로 실정책 롤아웃** — 기존 직결 `main_act.py`와 성공률 비교 (분리 자체가 성능을 깎지 않는지). 첫 에피소드는 정지 버튼에 손 올리고 관찰
4. **Tailscale 설치 → 핫스팟 전환** — 실측 RTT 보고 jpeg_quality / consume_cap_steps 조정
5. 실전 전 체크: top 카메라 화면의 USB 케이블 제거, 노트북 방화벽 8000 인바운드 허용

## 실행 명령 (cmd 기준)

```cmd
:: 창 1 — 정책 서버 (테스트는 --mock, 실전은 랩 서버에서 실체크포인트로)
cd "C:\Users\kangk\OneDrive\바탕 화면\4-1\2. 캡디경진대회\so101-vla-pipeline"
set PYTHONIOENCODING=utf-8
C:\Users\kangk\anaconda3\envs\lerobot312\python.exe -m so101_pipeline.servers.act_policy_server --mock

:: 창 2 — 에이전트 (--dry_run 붙이면 토크/쓰기 없음)
cd "C:\Users\kangk\OneDrive\바탕 화면\4-1\2. 캡디경진대회\so101-vla-pipeline"
set PYTHONIOENCODING=utf-8
set LEROBOT_CALIBRATION_DIR=C:\Users\kangk\OneDrive\바탕 화면\4-1\2. 캡디2\capstone
C:\Users\kangk\anaconda3\envs\lerobot312\python.exe -m so101_pipeline.runtime.main_edge

:: 카메라 재연결 후에는 매핑 재확인 (인덱스 셔플)
C:\Users\kangk\anaconda3\envs\lerobot312\python.exe -m so101_pipeline.runtime.main_edge --identify

:: 에피소드 시작 / 정지
curl -X POST http://127.0.0.1:8000/command/voice -H "Content-Type: application/json" -d "{\"text\":\"pick the banana and place it in the green basket\"}"
curl -X POST http://127.0.0.1:8000/command/stop

:: 카메라 확인: 브라우저에서 http://127.0.0.1:8000/video_feed
:: 앱: 같은 WiFi에서 노트북IP:8000 입력
```
