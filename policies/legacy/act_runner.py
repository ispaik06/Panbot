import logging
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Optional

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device


@dataclass
class ActRunConfig:
    # LeRobot RobotConfig (예: SO101FollowerConfig)
    robot: object

    # HF repo id or local path
    policy_path: str

    # control fps
    fps: int = 30

    # run for this duration; None => until stop_condition or Ctrl+C
    duration_s: float | None = None

    # optional multitask prompt
    task: str | None = None

    # obs rename (robot->policy)
    rename_map: dict[str, str] = field(default_factory=dict)

    # optional dataset stats
    dataset_repo_id: str | None = None
    dataset_root: str | None = None

    # device: "auto"|"cpu"|"cuda"
    device: str = "auto"
    use_amp: bool = True

    # fail-safe
    failsafe_rest_s: float = 1.0
    failsafe_rest_value: float = 0.0


def _build_dataset_features(robot, teleop_action_processor, robot_observation_processor) -> dict:
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )


def _load_dataset_stats(cfg: ActRunConfig) -> dict | None:
    if not cfg.dataset_repo_id:
        return None
    meta = LeRobotDatasetMetadata(cfg.dataset_repo_id, root=cfg.dataset_root)
    return rename_stats(meta.stats, cfg.rename_map)


def _build_rest_action(robot, rest_value: float = 0.0) -> dict[str, float]:
    return {k: float(rest_value) for k in robot.action_features.keys()}


def _choose_device(device_str: str) -> str:
    d = (device_str or "auto").lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d in ("cpu", "cuda"):
        if d == "cuda" and not torch.cuda.is_available():
            logging.warning("[ACT] requested cuda but not available -> cpu")
            return "cpu"
        return d
    raise ValueError(f"device must be auto|cpu|cuda (got {device_str})")


def run_act_policy(cfg: ActRunConfig, *, stop_condition: Optional[Callable[[], bool]] = None) -> None:
    """
    Runs ACT policy on the robot.
    - stop_condition(): if provided and returns True => stop loop.
    - duration_s: if set => stop after duration.
    """
    device = _choose_device(cfg.device)

    # Build policy config from pretrained
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = cfg.policy_path
    # Force device/amp here
    policy_cfg.device = device
    policy_cfg.use_amp = bool(cfg.use_amp)

    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = _build_dataset_features(
        robot=robot,
        teleop_action_processor=teleop_action_processor,
        robot_observation_processor=robot_observation_processor,
    )
    dataset_stats = _load_dataset_stats(cfg)

    policy = make_policy(cfg=policy_cfg, ds_meta=SimpleNamespace(features=dataset_features), rename_map=cfg.rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    rest_action = None

    logging.info("[ACT] connect robot...")
    robot.connect()
    try:
        rest_action = _build_rest_action(robot, cfg.failsafe_rest_value)

        start = time.perf_counter()
        step = 0
        logging.info(
            "[ACT] START policy=%s device=%s amp=%s fps=%d duration=%s",
            cfg.policy_path, device, cfg.use_amp, cfg.fps, cfg.duration_s
        )

        while True:
            loop_start = time.perf_counter()

            if stop_condition is not None and stop_condition():
                logging.info("[ACT] stop_condition=True -> stop")
                break

            if cfg.duration_s is not None and (time.perf_counter() - start) >= float(cfg.duration_s):
                logging.info("[ACT] duration reached -> stop")
                break

            obs = robot.get_observation()

            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy_cfg.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy_cfg.use_amp,
                task=cfg.task,
                robot_type=robot.robot_type,
            )

            act_processed_policy = make_robot_action(action_values, dataset_features)
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))

            robot.send_action(robot_action_to_send)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(0.0, 1.0 / cfg.fps - dt_s))
            step += 1

    except KeyboardInterrupt:
        logging.info("[ACT] KeyboardInterrupt")
    except Exception:
        logging.exception("[ACT] ERROR while running policy")
        raise
    finally:
        # Fail-safe: send rest action for a short time
        try:
            if rest_action is not None and cfg.failsafe_rest_s and cfg.failsafe_rest_s > 0:
                end = time.perf_counter() + float(cfg.failsafe_rest_s)
                while time.perf_counter() < end:
                    robot.send_action(rest_action)
                    precise_sleep(0.05)
        except Exception:
            logging.exception("[ACT] failsafe rest failed")

        try:
            robot.disconnect()
        except Exception:
            logging.exception("[ACT] disconnect failed")

        logging.info("[ACT] DONE")
