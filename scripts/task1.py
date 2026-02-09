import json
import logging
import time
from pathlib import Path
from dataclasses import asdict, dataclass, field
from pprint import pformat

import zmq
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


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


@dataclass
class Task1Config:
    robot: RobotConfig
    # Control loop frequency used while waiting for trigger.
    fps: int = 30
    # Dict of target joint positions. Keys can be motor names or feature names.
    rest_action: dict[str, float] | None = field(default_factory=lambda: dict(DEFAULT_REST_ACTION))
    # Sequence of poses (list of dicts). Each dict uses motor names or feature names.
    initial_sequence: list[dict[str, float]] = field(default_factory=lambda: list(DEFAULT_INITIAL_SEQUENCE))
    return_sequence: list[dict[str, float]] = field(default_factory=lambda: list(DEFAULT_RETURN_SEQUENCE))
    # Fill missing joints from the current robot position.
    use_current_for_missing: bool = True
    # Time to move from the current pose to the next pose.
    ramp_time_s: float = 3.0
    # Interval between commands during the ramp.
    ramp_interval_s: float = 0.05
    # Seconds to hold at each pose after ramping.
    pose_hold_s: float = 1.0
    # Interval between repeated commands while holding.
    hold_interval_s: float = 0.25
    # After finishing return_sequence, move to rest and hold there.
    hold_after_return: bool = True
    # Seconds to hold after returning to rest. If None and hold_after_return=True, hold until Ctrl+C.
    hold_time_s: float | None = None
    # ZeroMQ subscriber address for trigger signal.
    zmq_trigger_sub: str = "tcp://127.0.0.1:5559"
    # Debounce/confirm: require N consecutive frames or T seconds of triggered=True.
    trigger_confirm_frames: int = 1
    trigger_confirm_s: float | None = None
    # Enforce sequence lengths for safety.
    enforce_sequence_lengths: bool = True
    initial_sequence_len: int = 5
    return_sequence_len: int = 5
    # Log actions sent to the robot.
    log_action: bool = True


def _normalize_action_keys(action: dict[str, float]) -> dict[str, float]:
    normalized = {}
    for key, value in action.items():
        feature = key if key.endswith(".pos") else f"{key}.pos"
        normalized[feature] = float(value)
    return normalized


def _apply_robot_defaults(robot_cfg: RobotConfig) -> None:
    if isinstance(robot_cfg, SO101FollowerConfig):
        if robot_cfg.id is None:
            robot_cfg.id = "my_awesome_follower_arm"
        if robot_cfg.calibration_dir is None:
            robot_cfg.calibration_dir = Path(
                "/home/user/.cache/huggingface/lerobot/calibration/robots/so101_follower"
            )
        if hasattr(robot_cfg, "port") and robot_cfg.port is None:
            robot_cfg.port = "/dev/ttyACM1"


def _validate_action_keys(action: dict[str, float], action_features: set[str]) -> None:
    unknown = set(action) - action_features
    if unknown:
        known = ", ".join(sorted(action_features))
        raise ValueError(f"Unknown action keys: {sorted(unknown)}. Expected subset of: {known}")


def _fill_missing_with_current(
    robot,
    action_features: set[str],
    target_action: dict[str, float],
    use_current_for_missing: bool,
) -> dict[str, float]:
    if use_current_for_missing and len(target_action) < len(action_features):
        observation = robot.get_observation()
        for key in action_features - set(target_action):
            target_action[key] = observation[key]
    return target_action


def _ramp_to_action(
    robot,
    action_features: set[str],
    target_action: dict[str, float],
    ramp_time_s: float,
    ramp_interval_s: float,
) -> None:
    if ramp_time_s <= 0:
        robot.send_action(target_action)
        return

    observation = robot.get_observation()
    start_action = {key: observation[key] for key in action_features}
    start_t = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start_t
        alpha = min(1.0, elapsed / ramp_time_s)
        ramp_action = {
            key: start_action[key] + alpha * (target_action[key] - start_action[key])
            for key in action_features
        }
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


def _run_sequence(
    robot,
    action_features: set[str],
    sequence: list[dict[str, float]],
    cfg: Task1Config,
) -> dict[str, float] | None:
    last_action = None
    for idx, pose in enumerate(sequence, start=1):
        target = _normalize_action_keys(pose)
        _validate_action_keys(target, action_features)
        target = _fill_missing_with_current(robot, action_features, target, cfg.use_current_for_missing)

        if cfg.log_action:
            logging.info("Sequence pose %d/%d: %s", idx, len(sequence), target)

        _ramp_to_action(
            robot=robot,
            action_features=action_features,
            target_action=target,
            ramp_time_s=cfg.ramp_time_s,
            ramp_interval_s=cfg.ramp_interval_s,
        )
        _hold_action(robot, target, cfg.pose_hold_s, cfg.hold_interval_s)
        last_action = target
    return last_action


def _parse_trigger_message(message: str) -> bool | None:
    raw = message.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = raw

    if isinstance(payload, dict) and "triggered" in payload:
        return bool(payload["triggered"])
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, (int, float)):
        return bool(payload)
    if isinstance(payload, str):
        text = payload.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _wait_for_latch(
    robot,
    hold_action: dict[str, float],
    cfg: Task1Config,
    trigger_sock: zmq.Socket,
    trigger_poller: zmq.Poller,
) -> None:
    last_triggered = False
    trigger_true_frames = 0
    trigger_true_start = None

    while True:
        loop_start = time.perf_counter()
        try:
            events = dict(trigger_poller.poll(timeout=0))
            if trigger_sock in events and events[trigger_sock] == zmq.POLLIN:
                msg = trigger_sock.recv_string(flags=zmq.NOBLOCK)
                parsed = _parse_trigger_message(msg)
                if parsed is not None:
                    last_triggered = parsed
        except zmq.Again:
            pass

        if last_triggered:
            if trigger_true_frames == 0:
                trigger_true_start = time.perf_counter()
            trigger_true_frames += 1

            frames_ready = cfg.trigger_confirm_frames <= 1 or trigger_true_frames >= cfg.trigger_confirm_frames
            time_ready = False
            if cfg.trigger_confirm_s is not None and cfg.trigger_confirm_s > 0:
                time_ready = (time.perf_counter() - trigger_true_start) >= cfg.trigger_confirm_s

            if frames_ready or time_ready:
                logging.info("Trigger confirmed. Running return sequence.")
                return
        else:
            trigger_true_frames = 0
            trigger_true_start = None

        robot.send_action(hold_action)
        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(0.0, 1 / cfg.fps - dt_s))


def _validate_sequences(cfg: Task1Config) -> None:
    if not cfg.initial_sequence:
        raise ValueError(
            "initial_sequence is empty. Provide 6 poses in Task1Config.initial_sequence (list of joint dicts)."
        )
    if not cfg.return_sequence:
        raise ValueError(
            "return_sequence is empty. Provide 5 poses in Task1Config.return_sequence (list of joint dicts)."
        )
    if cfg.enforce_sequence_lengths:
        if len(cfg.initial_sequence) != cfg.initial_sequence_len:
            raise ValueError(
                f"initial_sequence must have {cfg.initial_sequence_len} poses, got {len(cfg.initial_sequence)}."
            )
        if len(cfg.return_sequence) != cfg.return_sequence_len:
            raise ValueError(
                f"return_sequence must have {cfg.return_sequence_len} poses, got {len(cfg.return_sequence)}."
            )


def run_task(cfg: Task1Config) -> None:
    init_logging()
    _apply_robot_defaults(cfg.robot)
    if isinstance(cfg.robot, SO101FollowerConfig):
        logging.info("Using so101_follower port=%s id=%s", cfg.robot.port, cfg.robot.id)
    logging.info(pformat(asdict(cfg)))

    _validate_sequences(cfg)

    robot = make_robot_from_config(cfg.robot)

    zmq_ctx = zmq.Context.instance()
    trigger_sock = zmq_ctx.socket(zmq.SUB)
    trigger_sock.setsockopt(zmq.CONFLATE, 1)
    trigger_sock.setsockopt(zmq.LINGER, 0)
    trigger_sock.connect(cfg.zmq_trigger_sub)
    trigger_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    trigger_poller = zmq.Poller()
    trigger_poller.register(trigger_sock, zmq.POLLIN)
    logging.info("ZMQ trigger subscriber connected to %s", cfg.zmq_trigger_sub)

    robot.connect()
    try:
        action_features = set(robot.action_features.keys())

        if cfg.rest_action is None:
            rest_action = {name: 0.0 for name in action_features}
        else:
            rest_action = _normalize_action_keys(cfg.rest_action)
            _validate_action_keys(rest_action, action_features)
            rest_action = _fill_missing_with_current(robot, action_features, rest_action, cfg.use_current_for_missing)

        if cfg.log_action:
            logging.info("Rest action: %s", rest_action)

        _ramp_to_action(
            robot=robot,
            action_features=action_features,
            target_action=rest_action,
            ramp_time_s=cfg.ramp_time_s,
            ramp_interval_s=cfg.ramp_interval_s,
        )
        _hold_action(robot, rest_action, cfg.pose_hold_s, cfg.hold_interval_s)

        last_initial_action = _run_sequence(robot, action_features, cfg.initial_sequence, cfg)
        hold_action = last_initial_action if last_initial_action is not None else rest_action

        _wait_for_latch(robot, hold_action, cfg, trigger_sock, trigger_poller)

        _run_sequence(robot, action_features, cfg.return_sequence, cfg)

        _ramp_to_action(
            robot=robot,
            action_features=action_features,
            target_action=rest_action,
            ramp_time_s=cfg.ramp_time_s,
            ramp_interval_s=cfg.ramp_interval_s,
        )

        if cfg.hold_after_return:
            start = time.perf_counter()
            while True:
                robot.send_action(rest_action)
                precise_sleep(cfg.hold_interval_s)
                if cfg.hold_time_s is not None and (time.perf_counter() - start) >= cfg.hold_time_s:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        trigger_sock.close()


@parser.wrap()
def main(cfg: Task1Config):
    run_task(cfg)


if __name__ == "__main__":
    register_third_party_devices()
    main()
