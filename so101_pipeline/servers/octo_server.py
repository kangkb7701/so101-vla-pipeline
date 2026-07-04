import os
import sys
import time

# [Protocol C] JAX VRAM 독점 방지 및 TensorFlow GPU 비활성화
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import grpc
import numpy as np
import tensorflow as tf
import copy
from datetime import datetime
from PIL import Image as PILImage

# TensorFlow GPU 차단
tf.config.set_visible_devices([], 'GPU')

import jax
import jax.tree_util as jtu
from concurrent import futures
from octo.model.octo_model import OctoModel

from so101_pipeline.proto import vla_pb2, vla_pb2_grpc

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OctoServer")

# ===================================================================
# [DEBUG] Wrist camera ablation flag
# ===================================================================
# True로 두면 wrist token을 pad_mask=False로 마스킹하여 attention에서 제외.
# Wrist OOD가 모델 흔들림의 원인인지 빠르게 진단할 때 사용.
# 학습 시에는 wrist 있는 데이터로만 학습됐지만, Octo 사전학습이 wrist 없는
# 데이터셋(Bridge 등)도 포함하므로 pad_mask=False 처리 자체는 학습된 경로.
# ===================================================================
MASK_WRIST_FOR_DEBUG = False
# ===================================================================


class OctoService(vla_pb2_grpc.VLAServiceServicer):
    def __init__(self, checkpoint_path=None, step=None):
        if checkpoint_path is None:
            checkpoint_path = "hf://rail-berkeley/octo-small-1.5"
            logger.warning(f"⚠️ 경고: 기본 사전 학습 모델({checkpoint_path})을 로드 중입니다.")
            self.model = OctoModel.load_pretrained(checkpoint_path)
        else:
            logger.info(f"🤖 Loading Octo model from {checkpoint_path} (Step: {step})...")
            self.model = OctoModel.load_pretrained(checkpoint_path, step=step)
            
        self.dataset_name = list(self.model.dataset_statistics.keys())[0]
        self.rng = jax.random.PRNGKey(int(time.time()))

        # 수정 없이 원본 통계값 그대로 사용
        self.action_stats = self.model.dataset_statistics[self.dataset_name]["action"]

        logger.info(f"📊 Dataset statistics loaded for: '{self.dataset_name}'")
        logger.info(f"📊 Observation keys: {list(self.model.example_batch['observation'].keys())}")

        self.prev_images = {}
        self.prev_state = None
        # 첫 호출 여부 (Reset 후/서버 시작 후 첫 Predict). True면 mask=[False, True].
        self.is_first_call = True

        # ----------------------------------------------------------------
        # [디버그] 에피소드 시작 N 프레임 저장 — perception 가설 검증용
        # 카메라 raw frame + 모델이 실제로 입력받는 preprocessed frame 둘 다 PNG로 저장.
        # Reset 호출 시 카운터/episode_id 초기화되어 새 에피소드마다 새 폴더처럼 작동.
        # ----------------------------------------------------------------
        self.frame_save_count = 0
        self.MAX_FRAMES_TO_SAVE = 5
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.frame_save_dir = os.getenv("DEBUG_FRAME_DIR", os.path.join(_repo_root, "debug_frames"))
        os.makedirs(self.frame_save_dir, exist_ok=True)
        self.episode_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # VP-SAM3 fine-tuning used full-frame resize for image_primary.
        # Do not center-crop the top image here; crosshair/bbox geometry must stay
        # aligned with the training distribution.
        self.primary_crop_bbox = None

        logger.info("✅ Octo model loaded. Starting JAX warmup...")
        self._warmup()
        logger.info("🔥 Octo model warmup complete and ready to serve.")

    def _warmup(self):
        """서버 시작 시 모델의 example_batch 뼈대를 복제하여 안전하게 웜업합니다."""
        dummy_obs = jtu.tree_map(
            lambda x: np.zeros((1,) + x.shape[1:], dtype=x.dtype), 
            self.model.example_batch['observation']
        )
        dummy_obs["timestep_pad_mask"] = np.array([[True, True]], dtype=bool)
        
        dummy_task = self.model.create_tasks(texts=["warm up"])
        self.model.sample_actions(
            dummy_obs, 
            dummy_task, 
            # 조작해둔 통계값을 적용
            unnormalization_statistics=self.action_stats,
            rng=jax.random.PRNGKey(0)
        )

    def _save_debug_frames(self, raws: dict, preprocessed: dict, step_idx: int):
        """첫 N step의 raw/preprocessed frame을 PNG로 저장.
        파일명 형식: {episode_id}_step{NN}_{camera}_{raw|model}.png
        """
        for cam_name, raw in raws.items():
            path = os.path.join(
                self.frame_save_dir,
                f"{self.episode_id}_step{step_idx:02d}_{cam_name}_raw.png",
            )
            PILImage.fromarray(raw).save(path)
        for cam_name, prep in preprocessed.items():
            path = os.path.join(
                self.frame_save_dir,
                f"{self.episode_id}_step{step_idx:02d}_{cam_name}_model.png",
            )
            PILImage.fromarray(prep).save(path)
        logger.info(
            f"💾 Debug frames saved | step={step_idx} cameras={list(raws.keys())} "
            f"dir={self.frame_save_dir}"
        )

    def _preprocess_image(self, img_rgb_np: np.ndarray, camera_name: str) -> np.ndarray:
        """학습 파이프라인(dlimp.resize_image + ResizeImageWrapper)과 동일한 전처리.
        - 두 카메라 모두 lanczos3 + antialias로 target size에 resize
        - image_primary(top)는 VP overlay가 포함된 full-frame resize만 적용
        - image_wrist(front)도 resize만 적용

        target_size는 체크포인트마다 다르므로 example_batch에서 자동 추출:
        - example_batch['observation']['image_primary'].shape[-3:-1] → (H, W)
        - example_batch['observation']['image_wrist'].shape[-3:-1]   → (H, W)
        """
        obs_example = self.model.example_batch['observation']
        if camera_name == "front":
            h, w = obs_example["image_wrist"].shape[-3:-1]
        else:
            h, w = obs_example["image_primary"].shape[-3:-1]
        target_size = (int(h), int(w))

        img_tensor = tf.convert_to_tensor(img_rgb_np, dtype=tf.uint8)
        img_resized = tf.image.resize(
            img_tensor, target_size, method="lanczos3", antialias=True
        )
        img_uint8 = tf.cast(
            tf.clip_by_value(tf.round(img_resized), 0, 255), tf.uint8
        ).numpy()
        return img_uint8

    def Predict(self, request, context):
        start_time = time.time()
        try:
            # 1. 이미지 전처리 (Window=2 통일, 학습 파이프라인 동일 절차)
            # 클라이언트가 raw RGB bytes(H*W*3)를 보내므로 imdecode/cvtColor 불필요.
            images = {}
            debug_raws, debug_preps = {}, {}  # 첫 N step 디버그용 캡처
            should_save_debug = self.frame_save_count < self.MAX_FRAMES_TO_SAVE
            for img_data in request.images:
                if img_data.height <= 0 or img_data.width <= 0:
                    logger.warning(f"⚠️ Image '{img_data.camera_name}' missing height/width")
                    continue
                expected = img_data.height * img_data.width * 3
                if len(img_data.data) != expected:
                    logger.warning(
                        f"⚠️ Image '{img_data.camera_name}' data size mismatch: "
                        f"got {len(img_data.data)}, expected {expected} "
                        f"({img_data.height}x{img_data.width}x3)"
                    )
                    continue
                img_rgb_raw = np.frombuffer(img_data.data, dtype=np.uint8).reshape(
                    img_data.height, img_data.width, 3
                ).copy()  # frombuffer는 read-only view → TF/NumPy 안전성 위해 copy
                img_rgb = self._preprocess_image(img_rgb_raw, img_data.camera_name)

                camera_name = img_data.camera_name
                if should_save_debug:
                    debug_raws[camera_name] = img_rgb_raw
                    debug_preps[camera_name] = img_rgb

                if camera_name not in self.prev_images:
                    self.prev_images[camera_name] = img_rgb
                images[camera_name] = np.stack(
                    [self.prev_images[camera_name], img_rgb], axis=0
                )[np.newaxis, ...]
                self.prev_images[camera_name] = img_rgb

            # 디버그 frame 저장 (첫 N step만)
            if should_save_debug and debug_raws:
                self._save_debug_frames(debug_raws, debug_preps, self.frame_save_count)
                self.frame_save_count += 1

            if not images:
                return vla_pb2.PredictResponse(actions=[0]*7, action_dim=7, chunk_size=0)

            # 2. Observation 구성 (camera_name: top → image_primary, front → image_wrist)
            observation = jtu.tree_map(
                lambda x: np.zeros((1,) + x.shape[1:], dtype=x.dtype),
                self.model.example_batch['observation']
            )

            observation["image_primary"] = images["top"]
            observation["image_wrist"] = images["front"]

            # ----------------------------------------------------------------
            # [Mask Handling] HistoryWrapper 컨벤션 (gym_wrappers.py:111-117)
            # 첫 호출(Reset 직후): prev=current로 history padding → mask=[False, True]
            # 두 번째 호출부터: prev는 진짜 직전 frame → mask=[True, True]
            # ----------------------------------------------------------------
            if self.is_first_call:
                step_mask = np.array([[False, True]], dtype=bool)
            else:
                step_mask = np.array([[True, True]], dtype=bool)

            observation["timestep_pad_mask"] = step_mask
            observation["pad_mask_dict"]["image_primary"] = step_mask
            observation["pad_mask_dict"]["image_wrist"] = step_mask
            observation["pad_mask_dict"]["timestep"] = step_mask
            observation["timestep"] = np.array([[0, 1]], dtype=np.int32)

            # === [DEBUG] Wrist mask ablation ===
            # MASK_WRIST_FOR_DEBUG=True면 wrist를 attention에서 강제 제외.
            # pad_mask를 모두 False로 → transformer가 wrist token 무시.
            # 이미지도 zero로 채워 numerical 영향 최소화 (어차피 mask로 제외되지만 안전망).
            if MASK_WRIST_FOR_DEBUG:
                observation["pad_mask_dict"]["image_wrist"] = np.array([[False, False]], dtype=bool)
                observation["image_wrist"] = np.zeros_like(observation["image_wrist"])
                if self.is_first_call:
                    logger.warning("🧪 [DEBUG] WRIST MASKED — Transformer가 wrist token 무시 중")
            # === ===

            if request.state and "proprio" in observation:
                state_np = np.array(request.state, dtype=np.float32)
                if self.prev_state is None:
                    self.prev_state = state_np
                observation["proprio"] = np.stack([self.prev_state, state_np], axis=0)[np.newaxis, ...]
                observation["pad_mask_dict"]["proprio"] = step_mask
                self.prev_state = state_np

            # 첫 호출 처리 완료 표시 (다음 호출부터는 [True, True])
            self.is_first_call = False

            # 3. 태스크 생성 및 추론
            task = self.model.create_tasks(texts=[request.instruction])
            
            self.rng, rng = jax.random.split(self.rng)
            actions = self.model.sample_actions(
                observation,
                task,
                # 여기서도 조작해둔 통계값을 사용합니다.
                unnormalization_statistics=self.action_stats,
                rng=rng
            )
            raw_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            # [CHUNK] action_horizon=K chunk 전체를 flatten해서 client로 반환.
            # 기존: actions_np[0].tolist()  → chunk[0] 1 step만 반환 (chunk_size=1).
            # 변경: actions_np.flatten()    → chunk K*action_dim 통째로 반환.
            # 클라이언트가 K step을 100ms 간격으로 소비하면 추론 cycle(~167ms) 동안
            # control loop가 끊김 없이 10Hz로 굴러간다.
            actions_np = np.array(actions)[0]                  # shape (K, action_dim)
            chunk_size = int(actions_np.shape[0])
            flat_actions = actions_np.flatten().tolist()       # length = K * action_dim
            # flat_actions[:3], [3:6], [6]은 여전히 chunk[0]의 pos/rot/grip이라 아래 logger는 그대로 OK.

            latency = (time.time() - start_time) * 1000
            logger.info(f"🚀 추론 완료 | Latency: {latency:.2f}ms | 명령: {request.instruction}")
            logger.info(
                f"action pos={np.round(flat_actions[:3], 5).tolist()} "
                f"rot={np.round(flat_actions[3:6], 5).tolist()} "
                f"grip={flat_actions[6]:+.3f}"
            )

            return vla_pb2.PredictResponse(
                actions=flat_actions,
                action_dim=7,
                chunk_size=chunk_size,   # [CHUNK] model returns its trained action_horizon.
            )

        except Exception as e:
            logger.error(f"🚨 Inference Error: {e}")
            import traceback
            traceback.print_exc()
            return vla_pb2.PredictResponse(actions=[0]*7, action_dim=7, chunk_size=0)

    def Reset(self, request, context):
        """에피소드 시작 시 prev_images, prev_state, is_first_call 초기화 +
        디버그 frame 저장 카운터/episode_id 리셋."""
        self.prev_images = {}
        self.prev_state = None
        self.is_first_call = True
        self.frame_save_count = 0
        self.episode_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(
            f"🔄 Server state reset | new episode_id={self.episode_id} "
            f"(will save first {self.MAX_FRAMES_TO_SAVE} frames)"
        )
        return vla_pb2.ResetResponse(status="ok")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # SELECT MODEL
    # ------------------------------------------------------------------
    # Raw top+side SO101 fine-tuned Octo selected from the 8K validation comparison.
    # main_real2.py expects this model's gripper output as binary open intent.
    # 체크포인트는 repo에 포함되지 않음 — OCTO_CHECKPOINT 환경변수 또는 checkpoints/ 아래 배치
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_ckpt_path = os.getenv(
        "OCTO_CHECKPOINT",
        os.path.join(_repo_root, "checkpoints", "octo_fruit_raw_vision_frozen"),
    )
    target_step = int(os.getenv("OCTO_CHECKPOINT_STEP", "6999"))

    # base_ckpt_path = "/home/aivlab/kkb_capstone/checkpoints8_pickup_v2_50ep_unfrozen_260518/best_val"
    # target_step =1699

    # pretraion octo
    # target_step =None
    # base_ckpt_path = "hf://rail-berkeley/octo-small-1.5"

    # CP 1
    # base_ckpt_path = "/home/aivlab/kkb_capstone/checkpoints/octo-small-1.5-so101"
    # target_step = 27999
    # -------------------------------------------------------------------

    vla_pb2_grpc.add_VLAServiceServicer_to_server(
        OctoService(checkpoint_path=base_ckpt_path, step=target_step), 
        server
    )
    
    server.add_insecure_port('[::]:50051')
    logger.info("🚀 Octo VLA Server started on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()