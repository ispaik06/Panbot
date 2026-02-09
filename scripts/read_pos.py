"""
Read joint positions from a follower arm with torque disabled.

Example:
  python scripts/read_pos.py --robot.type=so101_follower --robot.port=/dev/ttyACM1
"""

import logging
import select
import sys
import time
from pathlib import Path
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any

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

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows environments
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]


def _default_robot_config() -> RobotConfig:
    return SO101FollowerConfig(port="/dev/ttyACM1", id="my_awesome_follower_arm")


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


@dataclass
class ReadPosConfig:
    robot: RobotConfig = field(default_factory=_default_robot_config)
    # Interval between prints in seconds.
    interval_s: float = 1.0
    # Max run duration in seconds. If None, runs until Ctrl+C.
    duration_s: float | None = None


def _format_joint_positions(obs: dict[str, Any]) -> str:
    joint_items = _extract_joint_positions(obs)
    if not joint_items:
        return ""
    return _format_joint_items(joint_items)


def _format_joint_items(joint_items: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.3f}" for k, v in sorted(joint_items.items()))


def _extract_joint_positions(obs: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}


def _disable_torque(robot) -> bool:
    if hasattr(robot, "bus") and hasattr(robot.bus, "disable_torque"):
        robot.bus.disable_torque()
        return True
    if hasattr(robot, "disable_torque"):
        robot.disable_torque()
        return True
    return False


def _read_pending_keys() -> list[str]:
    if not sys.stdin.isatty():
        return []
    keys: list[str] = []
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            break
        ch = sys.stdin.read(1)
        if not ch:
            break
        keys.append(ch)
    return keys


class _CbreakMode:
    def __init__(self) -> None:
        self._enabled = False
        self._old_settings = None

    def __enter__(self) -> bool:
        if not sys.stdin.isatty() or termios is None or tty is None:
            return False
        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._enabled = True
        return True

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._enabled or self._old_settings is None or termios is None:
            return
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)


@parser.wrap()
def read_pos(cfg: ReadPosConfig) -> None:
    init_logging()
    _apply_robot_defaults(cfg.robot)
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    torque_disabled = _disable_torque(robot)
    if not torque_disabled:
        logging.warning("Torque disable not supported for %s. Continuing with torque enabled.", robot)

    saved_positions: list[dict[str, float]] = []
    last_positions: dict[str, float] = {}

    def handle_key(ch: str) -> bool:
        nonlocal saved_positions, last_positions
        if ch in ("s", "S"):
            if not last_positions:
                print("[save] no joint positions to save yet.", flush=True)
                return False
            saved_positions.append(dict(last_positions))
            print(f"[save] saved #{len(saved_positions)}", flush=True)
            return False
        if ch in ("c", "C"):
            if not saved_positions:
                print("[clear] nothing to remove.", flush=True)
                return False
            saved_positions.pop()
            print(f"[clear] removed, remaining {len(saved_positions)}", flush=True)
            return False
        if ch in ("\n", "\r"):
            if not saved_positions:
                print("[saved] (empty)", flush=True)
            else:
                for idx, joints in enumerate(saved_positions, start=1):
                    joint_str = _format_joint_items(joints)
                    print(f"[saved {idx}] {joint_str}", flush=True)
            return True
        return False

    start = time.perf_counter()
    try:
        with _CbreakMode() as cbreak_enabled:
            if cbreak_enabled:
                print("Press 's' to save, 'c' to remove last, Enter to print & exit.", flush=True)
            else:
                print("Interactive keys disabled (non-tty).", flush=True)

            while True:
                loop_start = time.perf_counter()
                obs = robot.get_observation()
                last_positions = _extract_joint_positions(obs)
                joint_str = _format_joint_items(last_positions) if last_positions else ""
                if joint_str:
                    print(f"[joints] {joint_str}", flush=True)
                else:
                    print("[joints] (no joint positions in observation)", flush=True)

                if cbreak_enabled:
                    for ch in _read_pending_keys():
                        if handle_key(ch):
                            return

                if cfg.duration_s is not None and time.perf_counter() - start >= cfg.duration_s:
                    break

                dt_s = time.perf_counter() - loop_start
                precise_sleep(max(0.0, cfg.interval_s - dt_s))
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


def main() -> None:
    register_third_party_devices()
    read_pos()


if __name__ == "__main__":
    main()
