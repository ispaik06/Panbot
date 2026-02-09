import logging
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

from .act_runner import ActRunConfig, run_act_policy


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root (expected dict): {p}")
    return data


def build_robot_from_cfg(cfg: dict[str, Any]) -> SO101FollowerConfig:
    robot_cfg = cfg.get("robot", {}) or {}
    cams_cfg = cfg.get("cameras", {}) or {}
    idx = (cams_cfg.get("indices") or {})

    w = int(cams_cfg.get("width", 640))
    h = int(cams_cfg.get("height", 480))
    fps = int(cams_cfg.get("fps", 30))
    fourcc = str(cams_cfg.get("fourcc", "MJPG"))

    cams = {
        "right": OpenCVCameraConfig(index_or_path=int(idx["right"]), width=w, height=h, fps=fps, fourcc=fourcc),
        "left": OpenCVCameraConfig(index_or_path=int(idx["left"]), width=w, height=h, fps=fps, fourcc=fourcc),
        "global": OpenCVCameraConfig(index_or_path=int(idx["global"]), width=w, height=h, fps=fps, fourcc=fourcc),
        "wrist": OpenCVCameraConfig(index_or_path=int(idx["wrist"]), width=w, height=h, fps=fps, fourcc=fourcc),
    }

    robot = SO101FollowerConfig(
        port=str(robot_cfg.get("port", "/dev/ttyACM0")),
        id=str(robot_cfg.get("id", "my_awesome_follower_arm")),
        cameras=cams,
    )

    calib = robot_cfg.get("calibration_dir", None)
    if calib:
        robot.calibration_dir = str(calib)

    return robot


def run_policy_from_yaml(
    yaml_path: str | Path,
    policy_name: str,
    *,
    duration_s_override: float | None = None,
    stop_condition: Optional[Callable[[], bool]] = None,
) -> None:
    cfg = load_yaml(yaml_path)

    policies = cfg.get("policies", {}) or {}
    if policy_name not in policies:
        raise KeyError(f"policy '{policy_name}' not found in {yaml_path}. available={list(policies.keys())}")

    p = policies[policy_name] or {}
    runtime = cfg.get("runtime", {}) or {}

    robot_cfg = build_robot_from_cfg(cfg)

    fps = int(runtime.get("fps", 30))
    duration_default = runtime.get("duration_s_default", None)
    if duration_s_override is not None:
        duration_s = float(duration_s_override)
    else:
        duration_s = None if duration_default is None else float(duration_default)

    device = str(runtime.get("device", "auto"))
    use_amp = bool(runtime.get("use_amp", True))

    repo_id = str(p["repo_id"])
    task = p.get("task", None)
    rename_map = p.get("rename_map", {}) or {}
    dataset_repo_id = p.get("dataset_repo_id", None)

    logging.info("[POLICY] name=%s repo_id=%s fps=%d duration=%s", policy_name, repo_id, fps, duration_s)

    run_cfg = ActRunConfig(
        robot=robot_cfg,
        policy_path=repo_id,
        fps=fps,
        duration_s=duration_s,
        task=task,
        rename_map=rename_map,
        dataset_repo_id=dataset_repo_id,
        device=device,
        use_amp=use_amp,
    )
    run_act_policy(run_cfg, stop_condition=stop_condition)
