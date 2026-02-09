from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from Panbot.tasks.base_pose import normalize_action_keys


# ✅ task1.py에 있던 기본 자세(=rest_action)
DEFAULT_REST_ACTION: Dict[str, float] = {
    "shoulder_pan.pos": -9.298892988929879,
    "shoulder_lift.pos": -98.8125530110263,
    "elbow_flex.pos": 99.90880072959416,
    "wrist_flex.pos": 50.977060322854726,
    "wrist_roll.pos": 4.299947561615085,
    "gripper.pos": 0.2722940776038121,
}

DEFAULT_INITIAL_SEQUENCE: List[Dict[str, float]] = [
    {
        "shoulder_pan.pos": -51.119,
        "shoulder_lift.pos": -5.567,
        "elbow_flex.pos": 75.829,
        "wrist_flex.pos": -43.922,
        "wrist_roll.pos": 57.608,
        "gripper.pos": 80.903,
    },
    {
        "shoulder_pan.pos": -59.631,
        "shoulder_lift.pos": 24.182,
        "elbow_flex.pos": 42.935,
        "wrist_flex.pos": -44.007,
        "wrist_roll.pos": 54.302,
        "gripper.pos": 25.834,
    },
    {
        "shoulder_pan.pos": -42.782,
        "shoulder_lift.pos": -33.787,
        "elbow_flex.pos": 14.13,
        "wrist_flex.pos": 35.536,
        "wrist_roll.pos": 54.092,
        "gripper.pos": 25.834,
    },
    {
        "shoulder_pan.pos": -20.842,
        "shoulder_lift.pos": -3.612,
        "elbow_flex.pos": 14.675,
        "wrist_flex.pos": 21.05,
        "wrist_roll.pos": 23.924,
        "gripper.pos": 25.703,
    },
    {
        "shoulder_pan.pos": -20.14,
        "shoulder_lift.pos": 26.902,
        "elbow_flex.pos": -23.944,
        "wrist_flex.pos": 36.298,
        "wrist_roll.pos": -21.091,
        "gripper.pos": 25.768,
    },
]

DEFAULT_RETURN_SEQUENCE: List[Dict[str, float]] = [
    {
        "shoulder_pan.pos": -20.14,
        "shoulder_lift.pos": 26.902,
        "elbow_flex.pos": -23.944,
        "wrist_flex.pos": 36.298,
        "wrist_roll.pos": -21.091,
        "gripper.pos": 25.768,
    },
    {
        "shoulder_pan.pos": -20.842,
        "shoulder_lift.pos": -3.612,
        "elbow_flex.pos": 14.675,
        "wrist_flex.pos": 21.05,
        "wrist_roll.pos": 23.924,
        "gripper.pos": 25.703,
    },
    {
        "shoulder_pan.pos": -42.782,
        "shoulder_lift.pos": -33.787,
        "elbow_flex.pos": 14.13,
        "wrist_flex.pos": 35.536,
        "wrist_roll.pos": 54.092,
        "gripper.pos": 25.834,
    },
    {
        "shoulder_pan.pos": -59.631,
        "shoulder_lift.pos": 24.182,
        "elbow_flex.pos": 42.935,
        "wrist_flex.pos": -44.007,
        "wrist_roll.pos": 54.302,
        "gripper.pos": 25.834,
    },
    {
        "shoulder_pan.pos": -51.119,
        "shoulder_lift.pos": -5.567,
        "elbow_flex.pos": 75.829,
        "wrist_flex.pos": -43.922,
        "wrist_roll.pos": 57.608,
        "gripper.pos": 80.903,
    },
]


@dataclass
class Task1MotionConfig:
    """
    ✅ stepper(비블로킹)로 Task1을 수행하기 위한 설정
    - loop는 main_runtime에서 30Hz로 돌린다고 가정
    """
    fps: int = 30

    ramp_time_s: float = 3.0
    pose_hold_s: float = 1.0

    use_current_for_missing: bool = True

    initial_sequence: List[Dict[str, float]] = field(default_factory=lambda: list(DEFAULT_INITIAL_SEQUENCE))
    return_sequence: List[Dict[str, float]] = field(default_factory=lambda: list(DEFAULT_RETURN_SEQUENCE))

    enforce_sequence_lengths: bool = True
    initial_sequence_len: int = 5
    return_sequence_len: int = 5


class Mode(Enum):
    NONE = auto()
    INITIAL = auto()
    RETURN = auto()


class Phase(Enum):
    NONE = auto()
    RAMP = auto()
    HOLD = auto()
    DONE = auto()


class Task1MotionStepper:
    """
    ✅ 30Hz step 기반 Task1 모션
    - start_initial(): initial 수행 시작
    - start_return(): return 수행 시작
    - interrupt_to_return(): initial 수행 중 즉시 return으로 전환
    - step(): 1 tick에서 action 1회 send
    """

    def __init__(self, robot, cfg: Task1MotionConfig, action_features: Optional[set[str]] = None):
        self.robot = robot
        self.cfg = cfg
        self.action_features = action_features or set(robot.action_features.keys())

        self._validate_sequences()

        self.mode: Mode = Mode.NONE
        self.phase: Phase = Phase.NONE

        self._seq: List[Dict[str, float]] = []
        self._pose_idx: int = 0
        self._target: Optional[Dict[str, float]] = None

        self._ramp_start_t: float = 0.0
        self._hold_end_t: float = 0.0

        self._start_action_vec: Optional[np.ndarray] = None
        self._target_action_vec: Optional[np.ndarray] = None
        self._keys_sorted: List[str] = sorted(self.action_features)

        self._last_pose_action: Optional[Dict[str, float]] = None

    # ---------------- public API ----------------

    def start_initial(self):
        self.mode = Mode.INITIAL
        self._seq = self.cfg.initial_sequence
        self._pose_idx = 0
        self._enter_pose_ramp(from_current=True)
        logging.info("[TASK1] start_initial (stepper)")

    def start_return(self):
        self.mode = Mode.RETURN
        self._seq = self.cfg.return_sequence
        self._pose_idx = 0
        self._enter_pose_ramp(from_current=True)
        logging.info("[TASK1] start_return (stepper)")

    def interrupt_to_return(self):
        """
        ✅ initial 진행 중이면 즉시 return으로 전환
        """
        if self.mode != Mode.INITIAL:
            return
        logging.info("[TASK1] interrupt_to_return requested")
        self.mode = Mode.RETURN
        self._seq = self.cfg.return_sequence
        self._pose_idx = 0
        self._enter_pose_ramp(from_current=True)

    def is_initial_done(self) -> bool:
        return self.mode == Mode.INITIAL and self.phase == Phase.DONE

    def is_return_done(self) -> bool:
        return self.mode == Mode.RETURN and self.phase == Phase.DONE

    def get_last_pose_action(self) -> Optional[Dict[str, float]]:
        """
        initial 완료 시 마지막 포즈(action dict). hold에 쓰면 됨.
        """
        return self._last_pose_action

    def step(self, now: Optional[float] = None) -> None:
        """
        30Hz 루프에서 매 tick 호출.
        - 내부 상태에 따라 ramp 또는 hold를 수행하고 send_action 1회 실행
        """
        if self.mode == Mode.NONE or self.phase in (Phase.NONE, Phase.DONE):
            return

        if now is None:
            now = time.perf_counter()

        if self.phase == Phase.RAMP:
            self._step_ramp(now)
        elif self.phase == Phase.HOLD:
            self._step_hold(now)

    # ---------------- internals ----------------

    def _validate_sequences(self):
        if not self.cfg.initial_sequence:
            raise ValueError("initial_sequence is empty")
        if not self.cfg.return_sequence:
            raise ValueError("return_sequence is empty")
        if self.cfg.enforce_sequence_lengths:
            if len(self.cfg.initial_sequence) != int(self.cfg.initial_sequence_len):
                raise ValueError(f"initial_sequence must have {self.cfg.initial_sequence_len} poses")
            if len(self.cfg.return_sequence) != int(self.cfg.return_sequence_len):
                raise ValueError(f"return_sequence must have {self.cfg.return_sequence_len} poses")

    def _validate_action_keys(self, action: Dict[str, float]):
        unknown = set(action) - self.action_features
        if unknown:
            known = ", ".join(sorted(self.action_features))
            raise ValueError(f"Unknown action keys: {sorted(unknown)}. Expected subset of: {known}")

    def _fill_missing_with_current(self, target: Dict[str, float]) -> Dict[str, float]:
        if self.cfg.use_current_for_missing and len(target) < len(self.action_features):
            obs = self.robot.get_observation()
            for k in self.action_features - set(target):
                target[k] = float(obs[k])
        return target

    def _pose_to_target_action(self, pose: Dict[str, float]) -> Dict[str, float]:
        target = normalize_action_keys(pose)
        self._validate_action_keys(target)
        target = self._fill_missing_with_current(target)
        return target

    def _action_dict_to_vec(self, action: Dict[str, float]) -> np.ndarray:
        return np.array([float(action[k]) for k in self._keys_sorted], dtype=np.float32)

    def _vec_to_action_dict(self, vec: np.ndarray) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self._keys_sorted, vec.tolist())}

    def _enter_pose_ramp(self, from_current: bool):
        if self._pose_idx >= len(self._seq):
            self.phase = Phase.DONE
            logging.info("[TASK1] %s sequence done", self.mode.name)
            return

        pose = self._seq[self._pose_idx]
        target = self._pose_to_target_action(pose)

        obs = self.robot.get_observation()
        start_action = {k: float(obs[k]) for k in self.action_features} if from_current else (self._last_pose_action or target)

        # store
        self._target = target
        self._ramp_start_t = time.perf_counter()
        self.phase = Phase.RAMP

        self._start_action_vec = self._action_dict_to_vec(start_action)
        self._target_action_vec = self._action_dict_to_vec(target)

        logging.info("[TASK1] %s pose %d/%d -> RAMP",
                     self.mode.name, self._pose_idx + 1, len(self._seq))

        # ramp_time_s가 0이면 즉시 HOLD로 전환
        if float(self.cfg.ramp_time_s) <= 0:
            self.robot.send_action(target)
            self._enter_pose_hold()

    def _enter_pose_hold(self):
        assert self._target is not None
        self.phase = Phase.HOLD
        self._hold_end_t = time.perf_counter() + float(self.cfg.pose_hold_s)
        self._last_pose_action = dict(self._target)

        logging.info("[TASK1] %s pose %d/%d -> HOLD",
                     self.mode.name, self._pose_idx + 1, len(self._seq))

        # pose_hold_s가 0이면 바로 다음 pose로
        if float(self.cfg.pose_hold_s) <= 0:
            self._advance_pose()

    def _advance_pose(self):
        self._pose_idx += 1
        if self._pose_idx >= len(self._seq):
            self.phase = Phase.DONE
            logging.info("[TASK1] %s sequence DONE", self.mode.name)
            return
        self._enter_pose_ramp(from_current=True)

    def _step_ramp(self, now: float):
        assert self._start_action_vec is not None and self._target_action_vec is not None and self._target is not None

        T = float(self.cfg.ramp_time_s)
        if T <= 1e-6:
            # 방어
            self.robot.send_action(self._target)
            self._enter_pose_hold()
            return

        alpha = min(1.0, (now - self._ramp_start_t) / T)
        vec = self._start_action_vec + alpha * (self._target_action_vec - self._start_action_vec)
        action = self._vec_to_action_dict(vec)
        self.robot.send_action(action)

        if alpha >= 1.0:
            self._enter_pose_hold()

    def _step_hold(self, now: float):
        assert self._target is not None
        # hold 동안에는 target pose를 계속 refresh
        self.robot.send_action(self._target)

        if now >= self._hold_end_t:
            self._advance_pose()
