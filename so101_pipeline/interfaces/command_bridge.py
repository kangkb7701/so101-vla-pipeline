import json
from dataclasses import dataclass
from urllib import request
from urllib.error import URLError, HTTPError


@dataclass
class CommandBridgeConfig:
    enabled: bool = False
    endpoint: str = "http://127.0.0.1:8000/command/latest"
    timeout_s: float = 1.0


class UserCommandBridge:
    """Fetches user command text and converts it to task description."""

    def __init__(self, config: CommandBridgeConfig):
        self.config = config

    def resolve_task_description(self, fallback: str) -> str:
        if not self.config.enabled:
            return fallback
        text = self._fetch_latest_instruction_text()
        if not text:
            return fallback
        return self._interpret(text)

    def _fetch_latest_instruction_text(self):
        try:
            req = request.Request(self.config.endpoint, method="GET")
            with request.urlopen(req, timeout=self.config.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError, ValueError):
            return None

        instruction = payload.get("instruction") or {}
        text = instruction.get("text")
        if isinstance(text, str):
            text = text.strip()
        return text or None

    def _interpret(self, command_text: str) -> str:
        # 앱에서 들어온 명령은 그대로 main_real2/VLA 쪽으로 전달한다.
        # 예전 smoke-test용 정규화는 "cube"가 포함된 모든 문장을 pickup으로
        # 바꿔 push/pull/place 같은 실제 의도를 훼손했다.
        return command_text.strip()

