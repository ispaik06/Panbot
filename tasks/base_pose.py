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

    ✅ 추가: ramp_to_target(duration_s)
      - 현재 관측 pose -> target pose를 duration_s 동안 선형 보간해서 전송
      - 램프가 끝나면 자동으로 target 유지 모드로 전환
    """

    def __init__(self, robot, cfg: HoldConfig, action_features: Optional[set[str]] = None):
        self.robot = robot
        self.cfg = cfg
        self.enabled = False
        self._last_send_t = 0.0

        self.action_features = action_features or set(robot.action_features.keys())
        self.target_action: Optional[Dict[str, float]] = None

        # ---- ramp state
        self._ramp_active = False
        self._ramp_start_t = 0.0
        self._ramp_duration_s = 0.0
        self._ramp_from: Optional[Dict[str, float]] = None
        self._ramp_to: Optional[Dict[str, float]] = None

    # -----------------------------
    # public APIs
    # -----------------------------
    def set_target(self, action: Dict[str, float]):
        """
        target_action을 설정(유지용).
        """
        target = self._make_full_action(action)
        self.target_action = target
        logging.info("[BASE_POSE] target set (%d keys)", len(target))

    def ramp_to_target(self, duration_s: float, target_action: Optional[Dict[str, float]] = None):
        """
        ✅ 현재 pose -> (target_action or self.target_action) 로 duration_s 동안 보간 전송.
        - 비블로킹: tick()이 계속 호출될 때 진행됨.
        """
        if target_action is None:
            if self.target_action is None:
                raise ValueError("ramp_to_target: target_action is None and self.target_action is None")
            target = dict(self.target_action)
        else:
            target = self._make_full_action(target_action)

        # start pose: 현재 관측에서 target에 필요한 키만 뽑기
        obs = self.robot.get_observation()
        start = {k: float(obs.get(k, target[k])) for k in target.keys()}

        self._ramp_active = True
        self._ramp_start_t = time.perf_counter()
        self._ramp_duration_s = max(0.05, float(duration_s))
        self._ramp_from = start
        self._ramp_to = target

        # ramp 끝나면 유지해야 하므로 target_action도 target으로 맞춰둠
        self.target_action = target

        # 즉시 한 번 보내고 시작(체감 좋음)
        self._last_send_t = 0.0

        logging.info("[BASE_POSE] ramp start: duration_s=%.3f", self._ramp_duration_s)

    def ramp_to_target_blocking(self, duration_s: float, target_action: Optional[Dict[str, float]] = None):
        """
        ✅ 블로킹 버전: 내부에서 루프를 돌며 ramp가 끝날 때까지 진행.
        - ESC 눌렀을 때 "천천히 복귀하고 종료"에 특히 편함.
        """
        self.ramp_to_target(duration_s, target_action=target_action)
        self.enable()
        while self._ramp_active:
            loop_start = time.perf_counter()
            self.tick()
            self.sleep_to_fps(loop_start)

    def enable(self):
        self.enabled = True
        self._last_send_t = 0.0
        logging.info("[BASE_POSE] enabled")

    def disable(self):
        self.enabled = False
        logging.info("[BASE_POSE] disabled")

    def is_ramping(self) -> bool:
        return bool(self._ramp_active)

    def tick(self):
        """
        main loop에서 자주 호출.
        내부적으로 hold_interval마다만 send_action.
        ramp 중이면 보간값을 보내고, ramp가 끝나면 target 유지로 전환.
        """
        if not self.enabled:
            return
        if self.target_action is None:
            return

        now = time.perf_counter()
        if (now - self._last_send_t) < float(self.cfg.hold_interval_s):
            return

        # ---- ramp mode
        if self._ramp_active and self._ramp_from is not None and self._ramp_to is not None:
            t = now - self._ramp_start_t
            a = min(1.0, max(0.0, t / self._ramp_duration_s))

            act = {}
            for k, v1 in self._ramp_to.items():
                v0 = self._ramp_from.get(k, v1)
                act[k] = (1.0 - a) * float(v0) + a * float(v1)

            self.robot.send_action(act)
            self._last_send_t = now

            if a >= 1.0:
                self._ramp_active = False
                self._ramp_from = None
                self._ramp_to = None
                logging.info("[BASE_POSE] ramp done -> hold target")
            return

        # ---- hold mode
        self.robot.send_action(self.target_action)
        self._last_send_t = now

    def sleep_to_fps(self, loop_start_t: float):
        dt = time.perf_counter() - loop_start_t
        period = 1.0 / max(int(self.cfg.fps), 1)
        precise_sleep(max(0.0, period - dt))

    # -----------------------------
    # internal helpers
    # -----------------------------
    def _make_full_action(self, action: Dict[str, float]) -> Dict[str, float]:
        target = normalize_action_keys(action)
        self._validate_action_keys(target)

        if self.cfg.use_current_for_missing and len(target) < len(self.action_features):
            obs = self.robot.get_observation()
            for k in (self.action_features - set(target)):
                target[k] = float(obs[k])
        return target

    def _validate_action_keys(self, action: Dict[str, float]):
        unknown = set(action) - self.action_features
        if unknown:
            known = ", ".join(sorted(self.action_features))
            raise ValueError(f"Unknown action keys: {sorted(unknown)}. Expected subset of: {known}")
