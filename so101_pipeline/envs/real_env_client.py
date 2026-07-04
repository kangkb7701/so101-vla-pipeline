import os
import zmq
import numpy as np
import genesis as gs
from scipy.spatial.transform import Rotation as R

class RealRobotEnvClient:
    def __init__(self, target_ip="localhost", target_port=5555, urdf_path=None):
        self.context = zmq.Context()
        self.target_url = f"tcp://{target_ip}:{target_port}"  # send_go_home에서 fresh socket용
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.target_url)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)

        print(f"📡 [Client] Connected to Hardware Server at {target_ip}:{target_port}")

        if urdf_path is None:
            urdf_path = os.getenv(
                "ROBOT_URDF",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "so101_new_calib.urdf"),
            )

        # [CPU BACKEND] IK용 shadow_robot은 CPU에서 돌린다.
        # 자동선택은 CUDA를 잡지만, octo_server(JAX/GPU)와 GPU를 공유하면
        # .cpu() 동기화가 추론 thread 대기로 ~130ms씩 지연된다.
        # 5-DOF IK는 CPU에서 0.4ms 수준이라 CPU 백엔드가 모든 면에서 우세.
        gs.init(seed=0, precision="32", logging_level="warning", backend=gs.cpu)
        self.scene = gs.Scene(show_viewer=False) 
        self.shadow_robot = self.scene.add_entity(
            gs.morphs.URDF(file=urdf_path, fixed=True)
        )
        self.scene.build()
        self.ee_link = self.shadow_robot.get_link("gripper_link")

    def get_state(self):
        try:
            self.socket.send_json({"cmd": "get_state"})
            response = self.socket.recv_json()
            
            if response["status"] == "ok":
                q_real_deg = np.array(response["q_deg"])
                q_real_rad = np.deg2rad(q_real_deg)

                self.shadow_robot.set_dofs_position(q_real_rad)
                
                pos = self.ee_link.get_pos().detach().cpu().numpy()
                quat_wxyz = self.ee_link.get_quat().detach().cpu().numpy()
                quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
                rot_mat = R.from_quat(quat_xyzw).as_matrix()
                
                return {
                    'pos': pos,
                    'quat': quat_xyzw,
                    'rot_mat': rot_mat,
                    'q': q_real_rad,
                    'gripper_telemetry': response.get('gripper_telemetry', {}),
                }
        except zmq.Again:
            print("🚨 [Client] Timeout!")
            return None

    def step(self, q_target, gripper_target):
        if hasattr(q_target, "detach"):
            q_target = q_target.detach().cpu().numpy()
            
        # [수정] 모든 관절을 Radian -> Degree로 직접 변환 (연속 제어 대응)
        q_target_deg = np.rad2deg(q_target).tolist()
        gripper_val = float(gripper_target)

        try:
            self.socket.send_json({
                "cmd": "step",
                "q_target_deg": q_target_deg,
                "gripper_val": gripper_val
            })
            self.socket.recv_json()
        except zmq.Again:
            print("🚨 [Client] Step Timeout!")

    def send_go_home(self, timeout_ms=8000):
        """hardware_server에 홈자세 복귀 명령. 서버에서 interpolation으로 천천히 이동.

        [Robust 설계]
        기존 self.socket을 재사용하면 KeyboardInterrupt가 get_state()/step()의
        recv_json 도중에 발생한 경우 socket이 "expecting recv" broken state로
        남아 send가 실패한다 (ZMQ REQ socket의 strictly alternating 제약).

        그래서 항상 fresh REQ socket을 새로 만들어 사용. 기존 socket 상태와
        독립적이라 finally 블록에서도 안전하게 작동.
        """
        print(f"🏠 [Client] send_go_home: fresh socket으로 시도 (timeout={timeout_ms}ms)")
        fresh_socket = self.context.socket(zmq.REQ)
        fresh_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        fresh_socket.setsockopt(zmq.SNDTIMEO, 3000)  # send도 hang 방지
        fresh_socket.setsockopt(zmq.LINGER, 0)       # close 시 즉시 종료, pending msg 버림
        try:
            fresh_socket.connect(self.target_url)
            print(f"🏠 [Client] go_home 명령 전송 → {self.target_url}")
            fresh_socket.send_json({"cmd": "go_home"})
            response = fresh_socket.recv_json()
            status = response.get("status")
            print(f"🏠 [Client] 서버 응답: status={status}")
            return status == "home_ok"
        except zmq.Again:
            print("🚨 [Client] go_home Timeout (8초 내 응답 없음)")
            return False
        except zmq.ZMQError as e:
            print(f"🚨 [Client] go_home ZMQError: {e}")
            return False
        except Exception as e:
            print(f"🚨 [Client] go_home 실패: {type(e).__name__}: {e}")
            return False
        finally:
            try:
                fresh_socket.close()
            except Exception:
                pass

    def disconnect(self):
        self.socket.close()
        self.context.term()
