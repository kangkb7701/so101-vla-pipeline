import grpc
import numpy as np
from PIL import Image
from .base_agent import BaseAgent
import proto.vla_pb2 as vla_pb2
import proto.vla_pb2_grpc as vla_pb2_grpc

class RemoteAgent(BaseAgent):
    def __init__(self, target_ip="localhost", target_port=50051):
        """gRPC 서버에 연결하는 원격 에이전트"""
        self.backend_name = "octo"
        self.target = f"{target_ip}:{target_port}"
        self.channel = grpc.insecure_channel(self.target)
        self.stub = vla_pb2_grpc.VLAServiceStub(self.channel)
        print(f"📡 Remote Agent connected to {target_ip}:{target_port}")

    def predict(self, image, instruction, wrist_image=None, state=None, raw_image=None):
        """
        image: PIL Image (Primary, RGB)
        instruction: str
        wrist_image: PIL Image (Optional, Wrist, RGB)
        state: list of floats (Optional, Robot Joint Positions)
        """
        # 1. 이미지를 raw RGB bytes로 직렬화 (학습 데이터와 동일하게 lossless)
        # JPEG q=90 인코딩(~3-5ms) + cvtColor 두 번을 모두 제거 → 사이클당 ~15ms 단축.
        def to_image_proto(pil_img, camera_name):
            img_np = np.ascontiguousarray(np.array(pil_img))  # RGB uint8 (H, W, 3)
            h, w = img_np.shape[:2]
            return vla_pb2.Image(
                data=img_np.tobytes(),
                camera_name=camera_name,
                height=h,
                width=w,
            )

        # 2. gRPC 요청 구성 (LeRobot 수집 시 명명 그대로: top / front)
        image_list = [to_image_proto(image, "top")]

        if wrist_image:
            image_list.append(to_image_proto(wrist_image, "front"))

        request = vla_pb2.PredictRequest(
            images=image_list,
            instruction=instruction,
            state=state if state is not None else []
        )

        # 3. RPC 호출
        try:
            response = self.stub.Predict(request)
            # [CHUNK] 서버가 (chunk_size * action_dim) flatten된 array를 보냄.
            # 기존: shape (7,) 단일 action 반환.
            # 변경: shape (chunk_size, action_dim) 예: (4, 7) 반환.
            # main_real2.py의 ChunkBuffer는 (K, action_dim)을 가정.
            # (구 main_real.py는 단일 action을 가정하므로 chunk[0]만 사용하려면
            #  호출부에서 arr[0]로 squeeze 필요)
            arr = np.array(response.actions, dtype=np.float32)
            if response.chunk_size > 0 and response.action_dim > 0:
                expected = response.chunk_size * response.action_dim
                if arr.size == expected:
                    return arr.reshape(response.chunk_size, response.action_dim)
            return arr  # fallback: shape 정보 없으면 그대로
        except grpc.RpcError as e:
            print(f"🚨 gRPC Error: {e.code()} - {e.details()}")
            # [CHUNK] fallback도 (1, 7) 2D로 통일해 client에서 ndim 분기를 단순화
            return np.zeros((1, 7), dtype=np.float32)

    def wait_until_ready(self, timeout_s=15.0):
        """Wait until the gRPC channel is actually ready, not merely constructed."""
        try:
            grpc.channel_ready_future(self.channel).result(timeout=timeout_s)
            return True
        except grpc.FutureTimeoutError:
            print(f"🚨 gRPC channel not ready after {timeout_s:.1f}s: {self.target}")
            return False

    def reset(self, retries=3, ready_timeout_s=15.0, rpc_timeout_s=5.0, retry_sleep_s=1.0):
        """에피소드 시작 시 서버 prev_images/prev_state 초기화."""
        if not self.wait_until_ready(timeout_s=ready_timeout_s):
            return False

        for attempt in range(1, retries + 1):
            try:
                response = self.stub.Reset(vla_pb2.ResetRequest(), timeout=rpc_timeout_s)
                print(f"🔄 Agent reset: {response.status}")
                return True
            except grpc.RpcError as e:
                print(
                    f"🚨 gRPC Reset Error [{attempt}/{retries}]: "
                    f"{e.code()} - {e.details()}"
                )
                if attempt < retries:
                    import time
                    time.sleep(retry_sleep_s)
        return False

    def close(self):
        if hasattr(self, "channel"):
            self.channel.close()

    def __del__(self):
        self.close()
