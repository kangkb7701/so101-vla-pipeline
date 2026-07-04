from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def predict(self, image, instruction):
        """이미지와 명령어를 받아 로우 액션(7차원 토큰 등)을 반환합니다."""
        pass
