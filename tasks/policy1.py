from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

from lerobot.utils.robot_utils import precise_sleep


@dataclass
class Policy1Config:
    """
    지금은 스켈레톤(placeholder).
    - duration_s 동안만 실행되도록 기본 구현.
    - 나중에 여기 step() 안에서 실제 policy inference + action 생성하면 됨.
    """
    fps: int = 30
    duration_s: float = 10.0  # TODO: 나중에 실제 종료조건으로 교체 가능
    log_every_s: float = 2.0


class Policy1Runner:
    def __init__(self, robot, cfg: Policy1Config, action_features: Optional[set[str]] = None):
        self.robot = robot
        self.cfg = cfg
        self.action_features = action_features or set(robot.action_features.keys())

        self._start_t: Optional[float] = None
        self._last_log_t: float = 0.0
        self._done: bool = False

    def start(self):
        self._start_t = time.perf_counter()
        self._last_log_t = 0.0
        self._done = False
        logging.info("[POLICY1] start (duration=%.1fs)", self.cfg.duration_s)

    def is_done(self) -> bool:
        return bool(self._done)

    def step(self, frame_bgr=None) -> Dict[str, Any]:
        """
        frame_bgr는 현재는 안 쓰지만, 나중에 비전 기반 policy면 여기서 사용 가능.
        """
        if self._start_t is None:
            raise RuntimeError("Policy1Runner.start() 먼저 호출해야 합니다.")

        now = time.perf_counter()
        elapsed = now - self._start_t

        # ✅ 여기에서 실제 policy action 생성하면 됨
        # 지금은 "현재 자세 유지" 형태로 더미 동작(안전)
        obs = self.robot.get_observation()
        action = {k: float(obs[k]) for k in self.action_features}
        self.robot.send_action(action)

        # 로그 (너무 자주 찍지 않게)
        if (now - self._last_log_t) >= float(self.cfg.log_every_s):
            self._last_log_t = now
            logging.info("[POLICY1] running... t=%.1fs", elapsed)

        # 종료 조건(임시)
        if elapsed >= float(self.cfg.duration_s):
            self._done = True
            logging.info("[POLICY1] done (time up)")
        return {"elapsed": elapsed, "done": self._done}

    def sleep_to_fps(self, loop_start_t: float):
        dt = time.perf_counter() - loop_start_t
        period = 1.0 / max(int(self.cfg.fps), 1)
        precise_sleep(max(0.0, period - dt))
