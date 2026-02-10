from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from lerobot.utils.robot_utils import precise_sleep


def normalize_action_keys(action: Dict[str, float]) -> Dict[str, float]:
    """'shoulder_pan' -> 'shoulder_pan.pos' 로 통일"""
    out: Dict[str, float] = {}
    for k, v in action.items():
        kk = k if k.endswith(".pos") else f"{k}.pos"
        out[kk] = float(v)
    return out


@dataclass
class HoldConfig:
    fps: int = 30
    hold_interval_s: float = 0.25  # send_action 주기(너무 빠를 필요 없음)
    use_current_for_missing: bool = True


class BasePoseController:
    """
    기본자세(또는 임의 자세)를 '유지'하는 컨트롤러.
    - tick()을 main loop에서 계속 호출하면, hold_interval마다 send_action 해줍니다.
    - enable/disable로 제어권 충돌 방지 (policy 실행 중에는 disable)
    """

    def __init__(self, robot, cfg: HoldConfig, action_features: Optional[set[str]] = None):
        self.robot = robot
        self.cfg = cfg
        self.enabled = False
        self._last_send_t = 0.0

        # action feature set은 한 번만 뽑아두면 효율적
        self.action_features = action_features or set(robot.action_features.keys())
        self.target_action: Optional[Dict[str, float]] = None

    def set_target(self, action: Dict[str, float]):
        target = normalize_action_keys(action)
        self._validate_action_keys(target)
        if self.cfg.use_current_for_missing and len(target) < len(self.action_features):
            obs = self.robot.get_observation()
            for k in self.action_features - set(target):
                target[k] = float(obs[k])
        self.target_action = target
        logging.info("[BASE_POSE] target set (%d keys)", len(target))

    def enable(self):
        self.enabled = True
        self._last_send_t = 0.0
        logging.info("[BASE_POSE] enabled")

    def disable(self):
        self.enabled = False
        logging.info("[BASE_POSE] disabled")

    def tick(self):
        """
        main loop에서 자주 호출.
        내부적으로 hold_interval마다만 send_action.
        """
        if not self.enabled:
            return
        if self.target_action is None:
            return

        now = time.perf_counter()
        if (now - self._last_send_t) < float(self.cfg.hold_interval_s):
            return

        self.robot.send_action(self.target_action)
        self._last_send_t = now

    def sleep_to_fps(self, loop_start_t: float):
        dt = time.perf_counter() - loop_start_t
        period = 1.0 / max(int(self.cfg.fps), 1)
        precise_sleep(max(0.0, period - dt))

    def _validate_action_keys(self, action: Dict[str, float]):
        unknown = set(action) - self.action_features
        if unknown:
            known = ", ".join(sorted(self.action_features))
            raise ValueError(f"Unknown action keys: {sorted(unknown)}. Expected subset of: {known}")
