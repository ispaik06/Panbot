# Panbot/policies/common_policy_runner.py

from __future__ import annotations

import time
import logging
import threading
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device


def _build_dataset_features(robot, teleop_action_processor, robot_observation_processor) -> Dict[str, Dict]:
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


def _load_dataset_stats(dataset_repo_id: Optional[str], dataset_root: Any, rename_map: Dict[str, str]) -> Optional[dict]:
    if not dataset_repo_id:
        return None
    meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    return rename_stats(meta.stats, rename_map)


def run_pretrained_policy_shared_robot(
    *,
    robot,
    repo_id: str,
    fps: int = 30,
    duration_s: Optional[float] = None,
    task: Optional[str] = None,
    rename_map: Optional[Dict[str, str]] = None,
    dataset_repo_id: Optional[str] = None,
    dataset_root: Any = None,
    use_amp: bool = True,
    print_joints: bool = False,
    print_joints_every: int = 30,
    stop_event: Optional[threading.Event] = None,   # ✅ 추가
) -> None:
    """
    ✅ B정석: robot.connect()를 여기서 하지 않습니다.
    - main_runtime에서 connect된 robot 인스턴스를 그대로 받아서 policy 제어만 수행합니다.
    - duration_s=None이면 Ctrl+C까지 계속.
    """
    rename_map = rename_map or {}

    logging.info("[POLICY] load pretrained repo_id=%s", repo_id)
    policy_cfg = PreTrainedConfig.from_pretrained(repo_id)
    policy_cfg.pretrained_path = repo_id

    # processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = _build_dataset_features(
        robot=robot,
        teleop_action_processor=teleop_action_processor,
        robot_observation_processor=robot_observation_processor,
    )
    dataset_stats = _load_dataset_stats(dataset_repo_id, dataset_root, rename_map)

    policy = make_policy(cfg=policy_cfg, ds_meta=SimpleNamespace(features=dataset_features), rename_map=rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    start = time.perf_counter()
    step = 0
    dt = 1.0 / max(1, int(fps))

    logging.info("[POLICY] start fps=%d duration_s=%s use_amp=%s task=%s", fps, duration_s, use_amp, task)

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                logging.info("[POLICY] stop_event set -> break")
                break
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            if print_joints and (step % max(1, int(print_joints_every)) == 0):
                joint_items = {k: v for k, v in obs.items() if k.endswith(".pos")}
                if joint_items:
                    joint_str = " ".join(f"{k}={float(v):.3f}" for k, v in sorted(joint_items.items()))
                    logging.info("[POLICY joints] %s", joint_str)

            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy_cfg.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=(use_amp and policy_cfg.use_amp and torch.cuda.is_available()),
                task=task,
                robot_type=robot.robot_type,
            )

            act_processed_policy = make_robot_action(action_values, dataset_features)
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))

            robot.send_action(robot_action_to_send)

            step += 1
            if duration_s is not None and (time.perf_counter() - start) >= float(duration_s):
                break

            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        logging.info("[POLICY] KeyboardInterrupt")
        return

    logging.info("[POLICY] done (steps=%d)", step)
