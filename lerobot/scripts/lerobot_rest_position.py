#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Move a robot to a rest position.

Example (SO-101 follower):

```shell
lerobot-rest-position \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem58760431541 \
  --robot.id=black \
  --rest_action='{shoulder_pan: 0, shoulder_lift: 0, elbow_flex: 0, wrist_flex: 0, wrist_roll: 0, gripper: 50}'
```

Notes:
- `rest_action` keys can be motor names (e.g. `shoulder_pan`) or full feature names
  (e.g. `shoulder_pan.pos`). Missing keys can be filled from the current position.
- If `rest_action` is omitted, the script sends 0.0 for all joints (mid-range in
  normalized units).
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
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

SEQUENCE_POSES: dict[str, dict[str, float]] = {
    "dough_finish": {
        "shoulder_pan.pos": -20.22140221402215,
        "shoulder_lift.pos": -3.1382527565733653,
        "elbow_flex.pos": 9.98632010943912,
        "wrist_flex.pos": 6.457094307561604,
        "wrist_roll.pos": -3.1463030938647165,
        "gripper.pos": 19.469026548672566,
    },
    "dough_pick": {
        "shoulder_pan.pos": -53.35793357933579,
        "shoulder_lift.pos": -9.0754877014419,
        "elbow_flex.pos": 79.48016415868673,
        "wrist_flex.pos": -62.27697536108751,
        "wrist_roll.pos": 55.37493445201889,
        "gripper.pos": 19.469026548672566,
    },
    "dough_place": {
        "shoulder_pan.pos": -53.35793357933579,
        "shoulder_lift.pos": -9.0754877014419,
        "elbow_flex.pos": 79.48016415868673,
        "wrist_flex.pos": -62.27697536108751,
        "wrist_roll.pos": 55.37493445201889,
        "gripper.pos": 70.469026548672566,
    },
    "dough_place_lift": {
        "shoulder_pan.pos": -53.35793357933579,
        "shoulder_lift.pos": 50.0754877014419,
        "elbow_flex.pos": 79.48016415868673,
        "wrist_flex.pos": -62.27697536108751,
        "wrist_roll.pos": 55.37493445201889,
        "gripper.pos": 70.469026548672566,
    },
    "rest_position": dict(DEFAULT_REST_ACTION),
}

DEFAULT_SEQUENCE: list[str] = [
    "dough_finish",
    "dough_pick",
    "dough_place",
    "dough_place_lift",
    "rest_position",
]


@dataclass
class RestPositionConfig:
    robot: RobotConfig
    # Dict of target joint positions. Keys can be motor names or feature names.
    rest_action: dict[str, float] | None = field(default_factory=lambda: dict(DEFAULT_REST_ACTION))
    # Fill missing joints from the current robot position.
    use_current_for_missing: bool = True
    # Move through a predefined sequence of poses before holding.
    use_sequence: bool = True
    # Shortcut: go only to dough_finish and hold.
    dough: bool = False
    # Pose names to execute when use_sequence=True.
    sequence: list[str] = field(default_factory=lambda: list(DEFAULT_SEQUENCE))
    # Time to move from the current pose to the rest pose.
    ramp_time_s: float = 3.0
    # Interval between commands during the ramp.
    ramp_interval_s: float = 0.05
    # Keep commanding the target pose (prevents drift, keeps torque active).
    hold: bool = True
    # Seconds to hold. If None and hold=True, hold until Ctrl+C.
    hold_time_s: float | None = None
    # Interval between repeated commands while holding.
    hold_interval_s: float = 0.25
    # Seconds to wait after sending the action if hold=False.
    wait_s: float = 2.0
    # Log the final action sent to the robot.
    log_action: bool = True


def _normalize_action_keys(action: dict[str, float]) -> dict[str, float]:
    normalized = {}
    for key, value in action.items():
        feature = key if key.endswith(".pos") else f"{key}.pos"
        normalized[feature] = float(value)
    return normalized


def _ramp_to_action(
    robot: Robot,
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


@parser.wrap()
def rest_position(cfg: RestPositionConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    try:
        action_features = set(robot.action_features.keys())

        if cfg.dough:
            sequence = ["dough_finish"]
            use_sequence = True
        else:
            sequence = cfg.sequence
            use_sequence = cfg.use_sequence

        if use_sequence:
            if not sequence:
                raise ValueError("use_sequence=True but sequence is empty.")
            for name in sequence:
                if name not in SEQUENCE_POSES:
                    raise ValueError(f"Unknown sequence pose '{name}'. Available: {sorted(SEQUENCE_POSES)}")

                target = _normalize_action_keys(SEQUENCE_POSES[name])
                unknown = set(target) - action_features
                if unknown:
                    known = ", ".join(sorted(action_features))
                    raise ValueError(f"Unknown action keys: {sorted(unknown)}. Expected subset of: {known}")

                if cfg.use_current_for_missing and len(target) < len(action_features):
                    observation = robot.get_observation()
                    for key in action_features - set(target):
                        target[key] = observation[key]

                if cfg.log_action:
                    logging.info("Sequence pose '%s': %s", name, target)

                _ramp_to_action(
                    robot=robot,
                    action_features=action_features,
                    target_action=target,
                    ramp_time_s=cfg.ramp_time_s,
                    ramp_interval_s=cfg.ramp_interval_s,
                )

            rest_action = target
        else:
            if cfg.rest_action is None:
                rest_action = {name: 0.0 for name in action_features}
            else:
                rest_action = _normalize_action_keys(cfg.rest_action)

            unknown = set(rest_action) - action_features
            if unknown:
                known = ", ".join(sorted(action_features))
                raise ValueError(f"Unknown action keys: {sorted(unknown)}. Expected subset of: {known}")

            if cfg.use_current_for_missing and len(rest_action) < len(action_features):
                observation = robot.get_observation()
                for key in action_features - set(rest_action):
                    rest_action[key] = observation[key]

            if cfg.log_action:
                logging.info("Rest action: %s", rest_action)

            _ramp_to_action(
                robot=robot,
                action_features=action_features,
                target_action=rest_action,
                ramp_time_s=cfg.ramp_time_s,
                ramp_interval_s=cfg.ramp_interval_s,
            )

        if cfg.hold:
            start = time.perf_counter()
            while True:
                robot.send_action(rest_action)
                precise_sleep(cfg.hold_interval_s)
                if cfg.hold_time_s is not None and (time.perf_counter() - start) >= cfg.hold_time_s:
                    break
        elif cfg.wait_s > 0:
            precise_sleep(cfg.wait_s)
    finally:
        robot.disconnect()


def main():
    register_third_party_devices()
    rest_position()


if __name__ == "__main__":
    main()
