# 원격 구동 셋업 가이드 — 엣지 노트북 + 모바일 앱

로봇을 연구실 밖에서 돌리기 위한 구성이다. 노트북(엣지)이 로봇·카메라·앱을 전부 담당하고, 연구실 서버는 추론만 한다. 이 문서대로 하면 아무 노트북이나 현장 박스로 만들 수 있다.

```
[휴대폰 앱] --같은 핫스팟/WiFi--> [노트북: main_edge] --websocket(공인IP)--> [랩서버: act_policy_server(GPU)]
  명령/영상 (IP:8000)              로봇팔+카메라 USB 연결                     ACT 추론만
```

- 앱과 노트북은 **반드시 같은 네트워크**에 있어야 한다 (노트북은 핫스팟 뒤라 밖에서 못 들어옴)
- 노트북→서버는 노트북이 **밖으로 거는** 연결 하나뿐. 서버 주소만 알면 어느 망에서든 됨
- 정지 버튼은 앱→노트북 로컬 구간이라 인터넷이 죽어도 먹는다. 링크가 끊기면 노트북이 알아서 홀드→홈복귀

---

## 1. 준비물

- Windows 노트북 (아래는 Windows 기준. 리눅스면 포트명 `COM3`→`/dev/ttyACM0` 정도만 다름)
- SO-101 로봇팔(서보 6개, 시리얼-USB 어댑터), USB 카메라 2대(top, wrist)
- **캘리브레이션 파일** — 레포의 `calibration/my_follower.json`에 포함돼 있다 (우리 팀 로봇팔 전용). 팔을 재캘리브레이션하면 이 파일도 갱신해서 커밋할 것. 다른 팔의 파일을 쓰면 각도가 전부 어긋난다
- Anaconda (또는 Miniconda)

## 2. conda 환경 구축

```cmd
conda create -n lerobot312 python=3.12 -y
conda activate lerobot312
pip install "lerobot[dataset,feetech]"
pip uninstall -y opencv-python-headless
pip install "opencv-python>=4.9,<4.14"
pip install websockets msgpack fastapi uvicorn
```

주의사항 (전부 실제로 겪은 것):

- **Python은 반드시 3.12 이상.** lerobot 0.5.0+가 3.12를 요구하는데, 3.11 이하에서 설치하면 pip이 **에러 없이 조용히 0.4.4로 다운그레이드**하고, 그 버전에는 필요한 모듈이 없어서 나중에 import 에러가 난다.
- **opencv-python-headless 제거 후 full 버전 재설치는 필수.** lerobot이 headless를 깔아주는데, headless 빌드는 Windows MSMF 카메라 백엔드가 동작하지 않아 카메라가 안 열린다.

## 3. 레포 받기

```cmd
git clone https://github.com/kangkb7701/so101-vla-pipeline
cd so101-vla-pipeline
git checkout edge-deployment
```

## 4. 하드웨어 연결 및 카메라 확인

로봇팔과 카메라 2대를 USB로 연결한다 (허브 사용 가능 — 대역폭·전원 실측으로 문제없음 확인됨).

**시리얼 포트 확인**: 장치 관리자 → 포트(COM & LPT)에서 COM 번호 확인 (보통 COM3).

**카메라 매핑 — 카메라를 꽂을 때마다 해야 한다:**

```cmd
conda activate lerobot312
python -m so101_pipeline.runtime.main_edge --identify
```

인덱스별 미리보기 PNG가 저장된다. 파일을 열어보고 어느 인덱스가 top(위에서 내려다보는 뷰), 어느 것이 wrist(그리퍼 시점)인지 눈으로 확인해 입력하면 `edge_cameras.json`에 저장된다.

> **카메라 인덱스는 USB를 뽑았다 꽂을 때마다 뒤섞인다.** 노트북 내장 웹캠까지 섞여서 이전 매핑을 그대로 쓰면 정책이 내장 웹캠 영상을 보게 된다. 재연결했으면 반드시 `--identify`를 다시 돌릴 것.

## 5. 실행

cmd 창에서 (PowerShell이면 `set` 대신 `$env:이름="값"`):

```cmd
cd <레포 경로>
conda activate lerobot312
set PYTHONIOENCODING=utf-8
set LEROBOT_CALIBRATION_DIR=<레포 경로>\calibration
set ROBOT_PORT=COM3
set ACT_SERVER_URL=ws://203.230.252.22:40010
python -m so101_pipeline.runtime.main_edge --max_step_deg 12 --min_infer_period_s 0.4
```

이 네 줄이 다 떠야 정상이다:

```
bus connected on COM3; cameras up; dry_run=False
app backend: http://0.0.0.0:8000  (video: /video_feed)
waiting for app command. allowed tasks: ...
policy server connected: ws://203.230.252.22:40010
```

옵션 설명:
- `--max_step_deg 12` — 틱당 관절 이동 상한. 실정책이 최대 ~10°/틱을 내므로 기본값 6이면 동작이 깎인다
- `--min_infer_period_s 0.4` — 관측 업로드 최소 간격. 0이면 적응 모드(이전 응답이 오는 즉시 다음 관측 전송)로 링크가 허용하는 최대 빈도로 돈다. 정밀도가 아쉬우면 0을 먼저 시험할 것
- 청크 재생은 베이스라인(main_act)의 ACTTemporalEnsembler를 벽시계 기준으로 이식한 것: 현재 시점을 커버하는 **모든** 미만료 청크가 exp(-coeff·rank) 가중(오래된 계획이 더 무거움, coeff=0.01)으로 표결한다. 과거 계획이 예약한 그리퍼 개폐가 제때 집행되는 건 이 구조 덕분이므로 `--ensemble_chunks 1`(끔)로 돌리지 말 것
- `--dry_run` — 붙이면 토크를 안 켜고 서보에 아무것도 안 쓴다. **처음 셋업 검증은 반드시 이걸로**

## 6. 앱 연결

1. 노트북 방화벽 허용 (관리자 cmd, 최초 1회):
   ```cmd
   netsh advfirewall firewall add rule name="SO101 edge app" dir=in action=allow protocol=TCP localport=8000 profile=any
   ```
2. 앱을 쓸 휴대폰을 **노트북과 같은 네트워크**에 연결 (핫스팟이면 그 핫스팟에)
3. 노트북 IP 확인: `ipconfig | findstr IPv4` (핫스팟이면 172.20.10.x 같은 주소)
4. 앱에서 IP = 노트북 IP, 포트 = `8000` 입력
5. 확인 순서: 서버 상태 정상 → 카메라 스트리밍 → 바구니 명령 전송(노트북 로그에 `episode N start`) → 정지 버튼

앱이 연결 안 될 때:
- 폰 브라우저에서 `http://<노트북IP>:8000/health` — `{"status":"ok"}`가 보이면 망은 정상이고 앱 권한 문제
- **iPhone이면 설정 → 해당 앱 → "로컬 네트워크" 권한을 켜야 한다** (꺼져 있으면 "네트워크 연결상태가 좋지 않음" 오류)
- 앱과 노트북이 정말 같은 망인지 다시 확인. **다른 망에서는 공인IP를 넣어도 절대 안 된다** (노트북은 통신사 CGNAT 뒤라 인바운드 불가)
- 핫스팟 IP는 껐다 켜면 바뀐다 — `ipconfig`로 재확인

## 7. 로봇 없이 / 팔 안 움직이고 검증하기

**GPU·체크포인트 없이 전 구간 테스트** (mock 서버가 "현재 자세 유지" 청크를 반환):

```cmd
:: 창 1 — 가짜 정책 서버
python -m so101_pipeline.servers.act_policy_server --mock

:: 창 2 — 에이전트 (dry_run: 팔 안 움직임)
set ACT_SERVER_URL=ws://127.0.0.1:8765
python -m so101_pipeline.runtime.main_edge --dry_run

:: 창 3 — 명령 주입 (앱 대신)
curl -X POST http://127.0.0.1:8000/command/voice -H "Content-Type: application/json" -d "{\"text\":\"pick the banana and place it in the green basket\"}"
```

정상이면: 청크 순환 → 약 10초 후 `auto stop: stable chunk` → 성공 기록. 정지는 `curl -X POST http://127.0.0.1:8000/command/stop`.

실팔 첫 테스트는 **mock 서버 + dry_run 뺀 상태**로 한다. 팔이 뻣뻣해지고(토크 인가) 10초 제자리 유지 후 홈으로 부드럽게 돌아오면 통과. 실정책 첫 에피소드는 정지 버튼에 손 올리고 지켜볼 것.

## 8. 실행 중 로그 읽는 법

```
[edge 0050] chunk_age= 0.11s steps=50 holds=0 clamps=2 infer=11ms link=up
```

- `chunk_age` — 지금 재생 중인 계획이 몇 초 전 관측 기준인지. 유선 ~0.1s, 핫스팟 0.3~0.8s면 정상
- `holds` — 청크가 말라서 홀드한 횟수. 계속 0이어야 정상, 수십씩 쌓이면 링크 문제
- `clamps` — 안전 클램프 발동 누적. 에피소드 초반 몇 회는 정상
- `infer` — 서버 추론 시간. 11~20ms가 정상

링크가 끊기면: 남은 청크 재생 → 홀드 → 5초 이상 두절 시 자동 홈복귀 → 재접속되면 다음 명령 대기. 사람이 할 일 없음.

## 9. 자주 걸리는 함정 요약

| 증상 | 원인 / 해결 |
|---|---|
| `import lerobot` 관련 모듈 없음 | Python 3.11 이하에 설치돼 0.4.4로 다운그레이드됨 → 3.12 env 새로 만들기 |
| 카메라가 안 열림 | opencv-python-headless가 깔려 있음 → full 버전으로 교체 (2절) |
| `could not open port 'COM3': PermissionError` | 다른 프로세스(이전 에이전트)가 포트 점유 중 → 이전 창 Ctrl+C. 에이전트는 동시에 하나만 |
| 콘솔에 한글/특수문자 찍다가 프로세스 사망 (UnicodeEncodeError) | `set PYTHONIOENCODING=utf-8` 누락 |
| 정책이 이상한 화면을 봄 | 카메라 재연결 후 `--identify` 안 함 → 인덱스 셔플됨 |
| 팔이 덜덜 떨림 | 구버전 코드. `edge-deployment` 최신 pull (서보 P게인 설정 포함) |
| 카메라 프레임 타임아웃으로 에이전트 사망 | 구버전 코드. 최신은 놓친 프레임을 직전 프레임으로 대체하고 `drops=`로 집계, 2초 연속 실패 시에만 안전정지 |
| 그리퍼가 닫힌 채 시작해 끝까지 안 열림 | 학습 데이터는 그리퍼 연 상태(~40)로 시작하는데 홈 자세는 닫아둠(2.3) → 닫힌 시작은 분포 밖이라 정책이 안 엶. 최신 코드는 에피소드 시작 시 자동으로 40까지 열어줌 (`--episode_start_gripper`) |
| place에서 그리퍼를 안 열고 복귀 | 구버전의 최신-청크 단독 재생 구조 문제: 정책이 열기를 "0.5초 뒤"로 계속 미루면 재생이 영원히 못 따라감. 최신 코드는 베이스라인 TE를 이식해 과거 계획의 열기 예약이 시각 도래 시 집행됨 |
| 그리퍼가 명령을 무시함 (`grip=` 명령/실측 괴리) | 서보 과부하 보호(발동 시 토크 25%로 제한) 가능성. 에이전트 끄고 `python scripts/gripper_diag.py`로 Status 확인, 걸려 있으면 전원 재인가 |
| 앱 "네트워크 연결상태가 좋지 않음" | 6절 하단 체크리스트 |

## 10. 랩 서버 쪽 (참고)

노트북 담당자는 몰라도 되지만, 서버가 내려가 있으면 랩에서:

```bash
python -m so101_pipeline.servers.act_policy_server \
  --port 40010 \
  --policy_path <ACT 체크포인트> \
  --dataset_repo_id <데이터셋 repo> --dataset_root <경로> \
  --device cuda
```

추가 의존성은 `pip install websockets msgpack`뿐. 서버는 무상태라 아무 때나 재시작해도 되고, 노트북이 알아서 재접속한다.

---

`mobile_app/mobile_app/`은 Flutter 앱 소스다. 앱 자체는 이 구조 전환에서 코드 수정이 전혀 없었고, 접속 IP만 노트북으로 입력하면 된다. 앱 빌드 방법은 `mobile_app/README.md`(Flutter 기본) 참고.
