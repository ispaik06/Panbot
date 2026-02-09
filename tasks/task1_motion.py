# Panbot/tasks/task1_motion.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep


# -------------------------
# Defaults (same spirit as your task1.py)
# -------------------------
DEFAULT_REST_ACTION: dict[str, float] = {
    "shoulder_pan.pos": -9.298892988929879,
    "shoulder_lift.pos": -98.8125530110263,
    "elbow_flex.pos": 99.90880072959416,
    "wrist_flex.pos": 50.977060322854726,
    "wrist_roll.pos": 4.299947561615085,
    "gripper.pos": 0.2722940776038121,
}

DEFAULT_INITIAL_SEQUENCE: list[dict[str, float]] = [
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

DEFAULT_RETURN_SEQUENCE: list[dict[str, float]] = [
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


# -------------------------
# Config
# -------------------------
@dataclass
class Task1MotionConfig:
    robot: RobotConfig

    # sequences
    rest_action: dict[str, float] | None = field(default_factory=lambda: dict(DEFAULT_REST_ACTION))
    initial_sequence: list[dict[str, float]] = field(default_factory=lambda: list(DEFAULT_INITIAL_SEQUENCE))
    return_sequence: list[dict[str, float]] = field(default_factory=lambda: list(DEFAULT_RETURN_SEQUENCE))

    # behavior
    use_current_for_missing: bool = True

    # ramp
    ramp_time_s: float = 3.0
    ramp_interval_s: float = 0.05

    # hold
    pose_hold_s: float = 1.0
    hold_interval_s: float = 0.25  # ✅ 유지(send_action) 주기

    # safety
    enforce_sequence_lengths: bool = True
    initial_sequence_len: int = 5
    return_sequence_len: int = 5

    log_action: bool = True


# -------------------------
# Helpers
# -------------------------
def _normalize_action_keys(action: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in action.items():
        feature = key if key.endswith(".pos") else f"{key}.pos"
        normalized[feature] = float(value)
    return normalized


def _apply_robot_defaults(robot_cfg: RobotConfig) -> None:
    """
    task1.py 기본값 느낌 그대로 유지:
    - so101_follower면 id/calibration_dir/port 기본값 채움
    """
    if isinstance(robot_cfg, SO101FollowerConfig):
        if robot_cfg.id is None:
            robot_cfg.id = "my_awesome_follower_arm"
        if robot_cfg.calibration_dir is None:
            robot_cfg.calibration_dir = Path(
                "/home/user/.cache/huggingface/lerobot/calibration/robots/so101_follower"
            )
        if hasattr(robot_cfg, "port") and robot_cfg.port is None:
            robot_cfg.port = "/dev/ttyACM0"  # ✅ 요청대로 기본 /dev/ttyACM0


def _validate_action_keys(action: dict[str, float], action_features: set[str]) -> None:
    unknown = set(action) - action_features
    if unknown:
        known = ", ".join(sorted(action_features))
        raise ValueError(f"Unknown action keys: {sorted(unknown)}. Expected subset of: {known}")


def _fill_missing_with_current(robot, action_features: set[str], target_action: dict[str, float], use_current: bool) -> dict[str, float]:
    if use_current and len(target_action) < len(action_features):
        obs = robot.get_observation()
        for key in action_features - set(target_action):
            target_action[key] = obs[key]
    return target_action


def _ramp_to_action(robot, action_features: set[str], target_action: dict[str, float], ramp_time_s: float, ramp_interval_s: float) -> None:
    if ramp_time_s <= 0:
        robot.send_action(target_action)
        return

    obs = robot.get_observation()
    start_action = {k: obs[k] for k in action_features}
    start_t = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start_t
        alpha = min(1.0, elapsed / ramp_time_s)
        ramp_action = {k: start_action[k] + alpha * (target_action[k] - start_action[k]) for k in action_features}
        robot.send_action(ramp_action)
        if alpha >= 1.0:
            break
        precise_sleep(ramp_interval_s)


def _hold_action(robot, action: dict[str, float], hold_s: float, hold_interval_s: float) -> None:
    if hold_s <= 0:
        return
    end_time = time.perf_counter() + hold_s
    while time.perf_counter() < end_time:
        robot.send_action(action)
        precise_sleep(hold_interval_s)


def _validate_sequences(cfg: Task1MotionConfig) -> None:
    if not cfg.initial_sequence:
        raise ValueError("initial_sequence is empty.")
    if not cfg.return_sequence:
        raise ValueError("return_sequence is empty.")

    if cfg.enforce_sequence_lengths:
        if len(cfg.initial_sequence) != cfg.initial_sequence_len:
            raise ValueError(f"initial_sequence must have {cfg.initial_sequence_len} poses, got {len(cfg.initial_sequence)}.")
        if len(cfg.return_sequence) != cfg.return_sequence_len:
            raise ValueError(f"return_sequence must have {cfg.return_sequence_len} poses, got {len(cfg.return_sequence)}.")


def _run_sequence(robot, action_features: set[str], sequence: list[dict[str, float]], cfg: Task1MotionConfig) -> Optional[dict[str, float]]:
    last_action = None
    for idx, pose in enumerate(sequence, start=1):
        target = _normalize_action_keys(pose)
        _validate_action_keys(target, action_features)
        target = _fill_missing_with_current(robot, action_features, target, cfg.use_current_for_missing)

        if cfg.log_action:
            logging.info("[TASK1] pose %d/%d: %s", idx, len(sequence), target)

        _ramp_to_action(robot, action_features, target, cfg.ramp_time_s, cfg.ramp_interval_s)
        _hold_action(robot, target, cfg.pose_hold_s, cfg.hold_interval_s)
        last_action = target
    return last_action


# -------------------------
# Controller (B method: keep holding in main loop)
# -------------------------
class Task1Controller:
    """
    메인 루프(main_runtime)에서:
      - start() 한 번 호출
      - WAIT_* 동안 hold_tick() 주기적으로 호출 (자세 유지)
      - YOLO trigger 시 do_return() 호출
    """

    def __init__(self, cfg: Task1MotionConfig):
        self.cfg = cfg
        _apply_robot_defaults(self.cfg.robot)
        _validate_sequences(self.cfg)

        self.robot = make_robot_from_config(self.cfg.robot)
        self.action_features: set[str] = set()
        self.rest_action: Optional[dict[str, float]] = None
        self.hold_action: Optional[dict[str, float]] = None

        self._last_hold_send_t = 0.0

    def connect(self):
        logging.info("[TASK1] connecting robot...")
        self.robot.connect()
        self.action_features = set(self.robot.action_features.keys())
        logging.info("[TASK1] connected. action_features=%d", len(self.action_features))

        # build rest_action
        if self.cfg.rest_action is None:
            rest = {name: 0.0 for name in self.action_features}
        else:
            rest = _normalize_action_keys(self.cfg.rest_action)
            _validate_action_keys(rest, self.action_features)
            rest = _fill_missing_with_current(self.robot, self.action_features, rest, self.cfg.use_current_for_missing)
        self.rest_action = rest

    def disconnect(self):
        logging.info("[TASK1] disconnecting robot...")
        try:
            self.robot.disconnect()
        except Exception:
            logging.exception("[TASK1] disconnect error")

    def start(self):
        """
        Task1 시작 동작:
          1) rest로 램프 이동
          2) initial_sequence 실행
          3) 마지막 포즈를 hold_action으로 설정
        """
        if self.rest_action is None:
            raise RuntimeError("connect() must be called before start().")

        logging.info("[TASK1] moving to rest...")
        _ramp_to_action(self.robot, self.action_features, self.rest_action, self.cfg.ramp_time_s, self.cfg.ramp_interval_s)
        _hold_action(self.robot, self.rest_action, self.cfg.pose_hold_s, self.cfg.hold_interval_s)

        logging.info("[TASK1] running initial sequence...")
        last = _run_sequence(self.robot, self.action_features, self.cfg.initial_sequence, self.cfg)
        self.hold_action = last if last is not None else self.rest_action

        logging.info("[TASK1] start done. hold_action set.")
        self._last_hold_send_t = 0.0  # reset tick timer

    def hold_tick(self):
        """
        ✅ 정석(B) 유지 방식:
        main loop에서 자주 호출하되, 실제 send_action은 hold_interval_s마다만 수행.
        """
        if self.hold_action is None:
            return
        now = time.perf_counter()
        if (now - self._last_hold_send_t) >= float(self.cfg.hold_interval_s):
            self.robot.send_action(self.hold_action)
            self._last_hold_send_t = now

    def do_return(self):
        """
        YOLO trigger 시 실행:
          1) return_sequence 실행
          2) rest로 복귀
          3) 이후 hold_action = rest로 바꿈 (계속 유지 가능)
        """
        if self.rest_action is None:
            raise RuntimeError("connect() must be called before do_return().")

        logging.info("[TASK1] RETURN sequence...")
        _run_sequence(self.robot, self.action_features, self.cfg.return_sequence, self.cfg)

        logging.info("[TASK1] back to rest...")
        _ramp_to_action(self.robot, self.action_features, self.rest_action, self.cfg.ramp_time_s, self.cfg.ramp_interval_s)
        _hold_action(self.robot, self.rest_action, self.cfg.pose_hold_s, self.cfg.hold_interval_s)

        self.hold_action = self.rest_action
        self._last_hold_send_t = 0.0
        logging.info("[TASK1] return done. holding rest.")
