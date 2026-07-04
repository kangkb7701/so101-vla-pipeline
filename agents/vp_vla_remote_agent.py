from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .base_agent import BaseAgent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VP_VLA_ROOT = PROJECT_ROOT / "VP-VLA"
if str(VP_VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(VP_VLA_ROOT))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy  # noqa: E402


VP_LANGUAGE_PREFIX = (
    "You are given two images: the first is the original robot observation, "
    "and the second has visual prompts overlaid highlighting the target object "
    "and target location. "
)


class VPVLARemoteAgent(BaseAgent):
    """VP-VLA websocket client using raw top, overlay top, and raw side images."""

    def __init__(
        self,
        target_ip: str = "127.0.0.1",
        target_port: int = 10093,
        stats_path: str | Path | None = None,
        unnorm_key: str | None = None,
    ) -> None:
        self.backend_name = "vp_vla"
        if stats_path is None:
            raise ValueError("VP-VLA requires dataset_statistics.json via VP_VLA_STATS_PATH")
        self.stats_path = Path(stats_path).expanduser().resolve()
        if not self.stats_path.exists():
            raise FileNotFoundError(self.stats_path)
        statistics = json.loads(self.stats_path.read_text(encoding="utf-8"))
        if unnorm_key is None:
            if len(statistics) != 1:
                raise ValueError(
                    f"VP_VLA_UNNORM_KEY is required; available keys: {sorted(statistics)}"
                )
            unnorm_key = next(iter(statistics))
        if unnorm_key not in statistics:
            raise KeyError(f"Unknown VP-VLA unnorm key {unnorm_key!r}: {sorted(statistics)}")
        self.unnorm_key = unnorm_key
        self.action_stats = statistics[unnorm_key]["action"]
        self.client = WebsocketClientPolicy(host=target_ip, port=target_port)
        print(
            f"VP-VLA Agent connected to {target_ip}:{target_port} "
            f"stats={self.stats_path} unnorm_key={self.unnorm_key}"
        )

    @staticmethod
    def _as_rgb_array(image: Image.Image | np.ndarray | None, name: str) -> np.ndarray:
        if image is None:
            raise ValueError(f"VP-VLA requires {name}")
        array = np.asarray(image, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"{name} must be RGB HxWx3, got {array.shape}")
        return np.ascontiguousarray(array)

    def _unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        normalized = np.clip(np.asarray(normalized_actions, dtype=np.float32), -1.0, 1.0)
        q01 = np.asarray(self.action_stats["q01"], dtype=np.float32)
        q99 = np.asarray(self.action_stats["q99"], dtype=np.float32)
        mask = np.asarray(
            self.action_stats.get("mask", np.ones_like(q01, dtype=bool)),
            dtype=bool,
        )
        binary_indices = np.flatnonzero(~mask)
        if binary_indices.size:
            normalized[..., binary_indices] = np.where(
                normalized[..., binary_indices] < 0.5, 0.0, 1.0
            )
        return np.where(
            mask,
            0.5 * (normalized + 1.0) * (q99 - q01) + q01,
            normalized,
        ).astype(np.float32)

    def predict(
        self,
        image,
        instruction,
        wrist_image=None,
        state=None,
        raw_image=None,
    ) -> np.ndarray:
        del state
        raw_top = self._as_rgb_array(raw_image, "raw top image")
        overlay_top = self._as_rgb_array(image, "overlay top image")
        raw_side = self._as_rgb_array(wrist_image, "raw side image")
        response = self.client.predict_action(
            {
                "examples": [
                    {
                        "image": [raw_top, overlay_top, raw_side],
                        "lang": VP_LANGUAGE_PREFIX + instruction,
                    }
                ],
                "do_sample": False,
            }
        )
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", {}).get("message", "VP-VLA inference failed"))
        normalized = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)
        if normalized.ndim != 3 or normalized.shape[0] != 1 or normalized.shape[-1] != 7:
            raise ValueError(f"Unexpected VP-VLA action shape: {normalized.shape}")
        return self._unnormalize_actions(normalized[0])

    def reset(self, *args, **kwargs) -> bool:
        return True

    def close(self) -> None:
        self.client.close()

    def __del__(self):
        if hasattr(self, "client"):
            self.close()
