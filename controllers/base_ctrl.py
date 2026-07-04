from abc import ABC, abstractmethod

class BaseController(ABC):
    @abstractmethod
    def get_joint_targets(self, raw_action, current_state):
        """델타 액션과 현재 로봇 상태를 받아 목표 관절각을 계산합니다."""
        pass
