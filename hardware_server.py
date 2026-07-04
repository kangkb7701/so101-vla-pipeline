import zmq
import torch
import numpy as np
import time

from gripper_telemetry import read_gripper_telemetry
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

def main():
    print("🤖 [Hardware Server] SO-ARM100 전용 ZMQ 제어 서버 부팅 중...")

    robot_config = SOFollowerRobotConfig(
        port="/dev/ttyACM1",
        id="my_follower",
        use_degrees=True 
    )

    robot = SOFollower(robot_config)
    robot.connect() 
    print("✅ 로봇 연결 및 토크 ON 완료!")

    joint_keys = [
        'shoulder_pan', 'shoulder_lift', 'elbow_flex',
        'wrist_flex', 'wrist_roll', 'gripper'
    ]
    # =========================================================
    print("\n[👀 물리적 초기 자세 (Torque ON 직후)]")
    init_obs = robot.get_observation()
    for k in joint_keys:
        deg = float(init_obs[f"{k}.pos"])
        marker = "🔥" if "shoulder" in k else "  "
        print(f"{marker} {k:15}: {deg:>7.2f} 도")

    FREEZE_WRIST_ROLL = True
    WRIST_ROLL_NAME = 'wrist_roll'
    WRIST_ROLL_LOCK_DEG = float(init_obs[f"{WRIST_ROLL_NAME}.pos"])
    if FREEZE_WRIST_ROLL:
        print(f"🔒 wrist_roll 하드웨어 freeze: {WRIST_ROLL_LOCK_DEG:.2f} 도로 고정")
    print("="*40 + "\n")
    # =========================================================

    print("\n" + "🔒"*25)
    print("🔒 [관절별 안전 제어 범위 (Safety Limits)]")
    
    safe_limits = {
        'shoulder_pan':  (-73.0,  62.0),
        'shoulder_lift': (-103.0, 92.0),
        'elbow_flex':    (-95.0,  96.0),
        'wrist_flex':    (0,  94.0),
        'wrist_roll':    (-165.0, -70.0),
        #  기존값
        # 'wrist_roll':    (-165.0, -4.0), 
        # gripper: 학습 데이터의 max=34.8 (mean peak=22.9)이라 96은 OOD 누적.
        # 학습 분포 max + 약간 여유로 40 상한. 추론 시 모델의 gripper 누적이
        # 학습 분포 안에 머무르도록 강제.
        'gripper':       (0.0,    40.0),
    }
    if FREEZE_WRIST_ROLL:
        lo, hi = safe_limits[WRIST_ROLL_NAME]
        safe_limits[WRIST_ROLL_NAME] = (min(lo, WRIST_ROLL_LOCK_DEG), max(hi, WRIST_ROLL_LOCK_DEG))

    for name, (lo, hi) in safe_limits.items():
        if name == 'gripper':
            print(f" - {name:15}: {lo} ~ {hi} (raw unit)")
        else:
            print(f" - {name:15}: {lo:>6.1f} 도  ~  {hi:>6.1f} 도")
            
    print("🔒"*25 + "\n")

    SLOW_ACCEL = 30
    print(f"🐌 하드웨어 가속도 제한: {SLOW_ACCEL}")
    robot.bus.configure_motors(maximum_acceleration=SLOW_ACCEL, acceleration=SLOW_ACCEL)
    time.sleep(1)

    # =========================================================
    # [홈자세] 실험 끝나면 자동 복귀할 목표 자세 (실측 로그 평균값)
    # safe_limits을 넘는 값은 limit 안쪽으로 살짝 조정해서 clip 발생 안 함
    # =========================================================
    HOME_DEG = {
        'shoulder_pan':    0.0,
        'shoulder_lift': -103.0,   # 실측 -104~-105 → safe_limit(-103) 안쪽
        'elbow_flex':     98.0,
        'wrist_flex':     74.5,
        'wrist_roll':   -165.0,    # 실측 -165~-167 → safe_limit(-165) 안쪽
        'gripper':        39.4,    # open; selected VP dataset start median action_first5 ~= 39.4
    }
    HOME_DURATION_S = 2.0          # 홈까지 이동 시간 (linear interpolation)
    HOME_HZ         = 20.0          # 홈 이동 보간 주파수

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("🚀 [ZMQ Server] Port 5555에서 명령 대기 중...\n")

    def go_home():
        """현재 자세에서 HOME_DEG까지 linear interpolation으로 천천히 이동.
        실험 종료 시 main_real2.py에서 'go_home' 명령으로 호출되거나,
        서버 자체가 Ctrl+C로 종료될 때 fallback으로 호출됨.
        """
        try:
            obs = robot.get_observation()
            start_q = {k: float(obs[f"{k}.pos"]) for k in joint_keys}
            n_steps = int(HOME_DURATION_S * HOME_HZ)
            print(f"🏠 홈자세 복귀 시작 ({HOME_DURATION_S:.1f}s, {n_steps} steps)")
            for i in range(n_steps + 1):
                alpha = i / n_steps
                action_dict = {}
                for name in joint_keys:
                    if FREEZE_WRIST_ROLL and name == WRIST_ROLL_NAME:
                        target = WRIST_ROLL_LOCK_DEG
                    else:
                        target = (1.0 - alpha) * start_q[name] + alpha * HOME_DEG[name]
                    min_l, max_l = safe_limits[name]
                    safe_val = float(np.clip(target, min_l, max_l))
                    action_dict[f"{name}.pos"] = torch.tensor(safe_val, dtype=torch.float32)
                robot.send_action(action_dict)
                time.sleep(1.0 / HOME_HZ)
            print("🏠 홈자세 복귀 완료")
            return True
        except Exception as e:
            print(f"⚠️ 홈 복귀 중 에러: {e}")
            return False

    try:
        while True:
            message = socket.recv_json()
            cmd = message.get("cmd")

            if cmd == "get_state":
                obs = robot.get_observation()
                q_deg = [float(obs[f"{k}.pos"]) for k in joint_keys]
                gripper_telemetry = read_gripper_telemetry(robot, obs)
                socket.send_json({
                    "status": "ok",
                    "q_deg": q_deg,
                    "gripper_telemetry": gripper_telemetry,
                })

            elif cmd == "step":
                q_target_deg = message.get("q_target_deg")
                gripper_val = message.get("gripper_val")

                action_dict = {}
                # print(f"[HW] Gripper 수신값: {gripper_val:.4f} | 범위: {safe_limits['gripper']}")

                for i, name in enumerate(joint_keys):
                    if name == 'gripper':
                        val = gripper_val
                    elif FREEZE_WRIST_ROLL and name == WRIST_ROLL_NAME:
                        val = WRIST_ROLL_LOCK_DEG
                    else:
                        val = q_target_deg[i]
                    min_l, max_l = safe_limits[name]
                    safe_val = np.clip(val, min_l, max_l)
                    action_dict[f"{name}.pos"] = torch.tensor(safe_val, dtype=torch.float32)
                    # hardware_server.py (수정 후 - 임시 바이패스)
                    # action_dict[f"{name}.pos"] = torch.tensor(val, dtype=torch.float32)
                    # print(f"[HW] {name} clip 후: {safe_val:.4f}")

                robot.send_action(action_dict)

                socket.send_json({"status": "moved"})

            elif cmd == "go_home":
                # main_real2.py의 finally에서 호출. interpolation이 끝날 때까지
                # ZMQ는 blocked. 클라이언트 측 RCVTIMEO는 HOME_DURATION_S 이상 여유 필요.
                ok = go_home()
                socket.send_json({"status": "home_ok" if ok else "home_failed"})

    except KeyboardInterrupt:
        print("\n🛑 서버 종료 신호 감지")
        # [Fallback] 서버 자체를 Ctrl+C로 끌 때도 자동 홈 복귀 시도.
        # main_real2.py가 이미 go_home 보냈으면 이중 호출이 되지만 idempotent라 안전.
        go_home()
    finally:
        if robot.is_connected:
            robot.disconnect()
        socket.close()
        context.term()
        print("🔌 연결 종료 완료")

if __name__ == "__main__":
    main()
