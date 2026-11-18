#!/usr/bin/env bash
# Bring up the lab-side (GPU) half of the demo stack.
#
# Split for the offsite demo:
#   venue  : Jetson Nano  -> jetson_camera_server.py  (cameras, :8080)
#            laptop       -> hardware_server.py       (robot serial, :5555)
#   lab    : this machine -> everything below
#
# Usage
#   scripts/run_lab_side.sh                 # all local, original behaviour
#   scripts/run_lab_side.sh --remote        # cameras + robot are elsewhere
#   scripts/run_lab_side.sh --remote --camera-host 100.x.y.z --robot-host 100.x.y.z
#   scripts/run_lab_side.sh --stop
#
# Each service goes to its own log under logs/demo/ and its PID under logs/demo/pids.
# --remote turns on JPEG transport and the inference floor, which is what makes the
# link fit inside a phone hotspot; see docs or check_demo_link.py for the numbers.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/src"
VP_VLA_DIR="$REPO_ROOT/VP-VLA"
SAM3_DIR="$VP_VLA_DIR/examples/Robocasa_tabletop/visual_prompt_utility"
LOG_DIR="$SRC_DIR/logs/demo"
PID_DIR="$LOG_DIR/pids"

CONDA="${CONDA_ROOT:-$HOME/anaconda3}/envs"
PY_VLA="$CONDA/vp_vla/bin/python"
PY_SAM="$CONDA/vp_sam/bin/python"

CKPT="${VP_VLA_CKPT:-$REPO_ROOT/outputs/train/vp_vla/so101_file000_005_ee7_vp_qwen_oft_fast3090/checkpoints/steps_008000_pytorch_model.pt}"
SAM3_MODEL="${SAM3_MODEL:-facebook/sam3}"   # the packaged default path does not exist locally

REMOTE=0
CAMERA_HOST="127.0.0.1"
ROBOT_HOST="localhost"
JPEG_QUALITY="${VP_VLA_WIRE_JPEG_QUALITY:-90}"
INFER_FLOOR="${MIN_INFER_INTERVAL_S:-0.33}"

while [ $# -gt 0 ]; do
  case "$1" in
    --remote)       REMOTE=1 ;;
    --camera-host)  CAMERA_HOST="$2"; shift ;;
    --robot-host)   ROBOT_HOST="$2"; shift ;;
    --jpeg-quality) JPEG_QUALITY="$2"; shift ;;
    --infer-floor)  INFER_FLOOR="$2"; shift ;;
    --stop)         MODE=stop ;;
    --status)       MODE=status ;;
    -h|--help)      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
MODE="${MODE:-start}"

mkdir -p "$LOG_DIR" "$PID_DIR"

# ------------------------------------------------------------------ helpers

port_open() { (echo >"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }

stop_all() {
  local found=0
  for f in "$PID_DIR"/*.pid; do
    [ -e "$f" ] || continue
    local name pid
    name="$(basename "$f" .pid)"
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "  stopping $name (pid $pid)"
      kill "$pid" 2>/dev/null
      found=1
    fi
    rm -f "$f"
  done
  [ "$found" = 1 ] || echo "  실행 중인 서비스 없음"
}

status_all() {
  printf '  %-14s %-8s %s\n' PORT STATE SERVICE
  for spec in "10093:VP-VLA policy" "10094:SAM3" "8010:main_real2 video" "8000:app_server" "5555:hardware_server"; do
    local port="${spec%%:*}" name="${spec#*:}"
    if port_open "$port"; then printf '  %-14s %-8s %s\n' "$port" "UP" "$name"
    else printf '  %-14s %-8s %s\n' "$port" "-" "$name"; fi
  done
}

# Wait until the port answers or the log shows it died. Model loads are slow
# (the policy checkpoint is ~10 GB), so this is worth the wait rather than
# racing ahead and reporting a confusing connection error downstream.
wait_for_port() {
  local port="$1" name="$2" log="$3" deadline=$((SECONDS + ${4:-900}))
  while [ $SECONDS -lt $deadline ]; do
    port_open "$port" && { echo "  $name  UP (:$port)"; return 0; }
    if grep -qE "Traceback|Error|error while attempting to bind|CUDA out of memory|Killed" "$log" 2>/dev/null; then
      echo "  $name  실패 — $log 확인"
      tail -5 "$log" | sed 's/^/      /'
      return 1
    fi
    sleep 2
  done
  echo "  $name  시간 초과 — $log 확인"
  return 1
}

launch() {
  local name="$1" workdir="$2"; shift 2
  local log="$LOG_DIR/$name.log" pidf="$PID_DIR/$name.pid"
  if [ -e "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "  $name  이미 실행 중 (pid $(cat "$pidf"))"
    return 0
  fi
  : > "$log"
  ( cd "$workdir" && exec "$@" >>"$log" 2>&1 ) &
  echo $! > "$pidf"
  echo "  $name  기동 중 (pid $!)  로그: ${log#$REPO_ROOT/}"
}

# ------------------------------------------------------------------ modes

case "$MODE" in
  stop)   echo "서비스 중지"; stop_all; exit 0 ;;
  status) echo "포트 상태"; status_all; exit 0 ;;
esac

for py in "$PY_VLA" "$PY_SAM"; do
  [ -x "$py" ] || { echo "파이썬 없음: $py  (CONDA_ROOT로 지정 가능)" >&2; exit 1; }
done
[ -f "$CKPT" ] || { echo "체크포인트 없음: $CKPT  (VP_VLA_CKPT로 지정 가능)" >&2; exit 1; }

if [ "$REMOTE" = 1 ]; then
  export VP_VLA_WIRE_JPEG=1
  export VP_VLA_WIRE_JPEG_QUALITY="$JPEG_QUALITY"
  export MIN_INFER_INTERVAL_S="$INFER_FLOOR"
  export USE_DIRECT_CAMERA_SOURCE=false
  export CAMERA_HOST="$CAMERA_HOST"
  export HARDWARE_SERVER_HOST="$ROBOT_HOST"
  echo "모드: REMOTE  (JPEG q$JPEG_QUALITY, 추론 최소간격 ${INFER_FLOOR}s)"
  echo "  카메라 호스트: $CAMERA_HOST:8080     로봇 호스트: $ROBOT_HOST:5555"
else
  echo "모드: LOCAL  (전부 이 머신, 기존 동작과 동일)"
fi
echo

echo "GPU 서비스 기동 (모델 로딩에 수 분 걸립니다)"
launch vp_vla_policy "$VP_VLA_DIR" \
  "$PY_VLA" deployment/model_server/server_policy.py --port 10093 --idle_timeout -1 --ckpt_path "$CKPT"
launch sam3 "$SAM3_DIR" \
  "$PY_SAM" sam3_server.py --model-path "$SAM3_MODEL" --port 10094 --idle-timeout -1

wait_for_port 10093 "VP-VLA policy" "$LOG_DIR/vp_vla_policy.log" || exit 1
wait_for_port 10094 "SAM3"          "$LOG_DIR/sam3.log"          || exit 1
echo

echo "앱 서버 기동"
launch app_server "$SRC_DIR" "$PY_VLA" app_server/main.py
wait_for_port 8000 "app_server" "$LOG_DIR/app_server.log" 60 || exit 1
echo

cat <<MSG
GPU 쪽 준비 완료.

이제 제어 루프를 포그라운드로 띄우세요 (로그를 직접 봐야 합니다):

  cd $SRC_DIR
  APP_VIDEO_SERVER_ENABLED=true USE_USER_COMMAND_TASK_BRIDGE=true \\
MSG
if [ "$REMOTE" = 1 ]; then
cat <<MSG
      VP_VLA_WIRE_JPEG=1 VP_VLA_WIRE_JPEG_QUALITY=$JPEG_QUALITY \\
      MIN_INFER_INTERVAL_S=$INFER_FLOOR \\
      USE_DIRECT_CAMERA_SOURCE=false CAMERA_HOST=$CAMERA_HOST \\
      HARDWARE_SERVER_HOST=$ROBOT_HOST \\
MSG
fi
cat <<MSG
      $PY_VLA main_real2.py

중지: scripts/run_lab_side.sh --stop      상태: scripts/run_lab_side.sh --status
MSG
