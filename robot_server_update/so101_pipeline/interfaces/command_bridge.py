import json
import re
from dataclasses import dataclass
from urllib import request
from urllib.error import URLError, HTTPError


# 데모 task 공간: 바나나를 세 바구니 중 하나에 넣는 것뿐이므로,
# 자유 발화에서 색상/물체/동작 키워드만 뽑아 학습된 표준 문장으로 재조립한다.
# 표준 문장은 VLA 학습 문장과 동일해야 하고 parse_pick_place_task 정규식도 통과한다.
CANONICAL_TASKS = {
    "green": "pick the banana and place it in the green basket",
    "yellow": "pick the banana and place it in the yellow basket",
    "blue": "pick the banana and place it in the blue basket",
}

COLOR_KEYWORDS = {
    "green": ("green", "초록", "녹색", "그린"),
    "yellow": ("yellow", "노란", "노랑", "노란색", "옐로"),
    "blue": ("blue", "파란", "파랑", "파란색", "블루", "푸른"),
}

# 색상 단어가 우연히 들어간 문장(예: "the sky is blue")을 명령으로 오인하지 않도록
# 물체 또는 동작 키워드가 함께 있어야 정규화한다.
OBJECT_KEYWORDS = ("banana", "바나나")
ACTION_KEYWORDS = (
    "pick", "place", "put", "move", "grab", "drop",
    "집", "넣", "옮겨", "옮기", "놓", "담", "가져",
)

# main_real2.parse_pick_place_task와 동일한 패턴 (여기서 import하면 순환 참조라 복사본 유지)
_PICK_PLACE_PATTERNS = (
    r"^pick(?:\s+up)?\s+(?P<object>.+?)\s+(?:and\s+)?place\s+(?:it|them|the\s+object)?\s*(?:in|into|on|onto|to)\s+(?P<location>.+)$",
    r"^pick(?:\s+up)?\s+(?P<object>.+?)\s+(?:and\s+|then\s+)?put\s+(?:it|them|the\s+object)?\s*(?:in|into|on|onto|to)\s+(?P<location>.+)$",
    r"^move\s+(?P<object>.+?)\s+(?:in|into|on|onto|to)\s+(?P<location>.+)$",
)


def normalize_command(command_text):
    """자유 발화(한국어/영어)를 표준 task 문장으로 정규화한다.

    해석할 수 없으면 None을 반환한다. 호출 측은 None이면 해당 명령을 무시하고
    다음 명령을 기다려야 한다 (프로그램 종료 금지).
    """
    text = (command_text or "").strip().lower()
    if not text:
        return None

    matched_colors = [
        color for color, keywords in COLOR_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]
    has_object = any(keyword in text for keyword in OBJECT_KEYWORDS)
    has_action = any(keyword in text for keyword in ACTION_KEYWORDS)

    # 색상이 정확히 하나 + 물체/동작 단서가 있으면 표준 문장으로 변환
    if len(matched_colors) == 1 and (has_object or has_action):
        return CANONICAL_TASKS[matched_colors[0]]

    # 키워드로 못 잡아도 이미 표준 pick/place 형식(다른 물체 포함)이면 그대로 통과
    cleaned = re.sub(r"[.?!]+$", "", re.sub(r"\s+", " ", text)).strip()
    for pattern in _PICK_PLACE_PATTERNS:
        if re.match(pattern, cleaned, flags=re.IGNORECASE):
            return command_text.strip()

    return None


@dataclass
class CommandBridgeConfig:
    enabled: bool = False
    endpoint: str = "http://127.0.0.1:8000/command/latest"
    timeout_s: float = 1.0


class UserCommandBridge:
    """Fetches user command text and converts it to task description."""

    def __init__(self, config: CommandBridgeConfig):
        self.config = config
        self._last_rejected = None

    def resolve_task_description(self, fallback: str) -> str:
        if not self.config.enabled:
            return fallback
        text = self._fetch_latest_instruction_text()
        if not text:
            return fallback
        normalized = normalize_command(text)
        if normalized is None:
            if text != self._last_rejected:
                self._last_rejected = text
                print(
                    f"⚠️ 해석할 수 없는 명령이라 무시합니다: '{text}' "
                    "(예: '바나나를 파란 바구니에 넣어줘' / 'put the banana in the blue basket')"
                )
            return fallback
        if normalized != text:
            print(f"🈯 명령 정규화: '{text}' → '{normalized}'")
        return normalized

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
