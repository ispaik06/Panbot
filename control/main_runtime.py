# Panbot/control/main_runtime.py
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import yaml

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots import make_robot_from_config
from lerobot.utils.utils import init_logging

from Panbot.vision.modules.camera import open_camera, resize_for_preview
from Panbot.vision.modules.yoloseg_infer import YOLOSegConfig, YOLOSegInfer
from Panbot.vision.modules.gru_infer import GRUInferConfig, GRUInfer

from Panbot.tasks.base_pose import BasePoseController, HoldConfig
from Panbot.tasks.task1_motion import Task1MotionConfig, Task1MotionStepper, DEFAULT_REST_ACTION
from Panbot.policies.common_policy_runner import run_pretrained_policy_shared_robot


# -----------------------------
# Defaults (fallback if yaml missing)
# -----------------------------
DEFAULT_BASE_POSE = dict(DEFAULT_REST_ACTION)  # task1 rest_action == base pose로 사용


@dataclass
class RuntimePaths:
    corners: Path
    yolo_model: Path
    gru_ckpt: Path


def _env_override_str(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key, None)
    return v if (v is not None and v != "") else default


def _env_override_int(key: str, default: int) -> int:
    v = os.environ.get(key, None)
    if v is None or v == "":
        return default
    return int(v)


def _env_override_float(key: str, default: float) -> float:
    v = os.environ.get(key, None)
    if v is None or v == "":
        return default
    return float(v)


def _setup_logging(log_dir: Path) -> None:
    """
    ✅ 로그는 덮어쓰기 아니라 쌓이도록(append)
    - 터미널 + 파일 동시 출력
    """
    init_logging()  # lerobot 기본 logging 초기화 (있으면 활용)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"main_runtime_{ts}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 이미 handler가 붙어있을 수 있으니 중복 방지
    for h in list(root.handlers):
        # init_logging이 붙인 handler 유지하되, 중복 스트림/파일은 정리
        pass

    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    # 중복 추가 방지: 같은 타입 handler 이미 있으면 스킵
    has_file = any(isinstance(h, logging.FileHandler) for h in root.handlers)
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    if not has_file:
        root.addHandler(file_handler)
    if not has_stream:
        root.addHandler(stream_handler)

    logging.info("==================================================")
    logging.info("[main_runtime] logging started")
    logging.info("log_file=%s", log_path)
    logging.info("==================================================")


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime.yaml must be a mapping(dict) at top-level")
    return data


def _require_file(p: Path, name: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"missing {name}: {p}")


def _read_runtime_config(yaml_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(yaml_path)

    # ✅ 환경변수 override (start_all.sh에서 넣기 쉬움)
    # 예: PANBOT_ROBOT_PORT=/dev/ttyACM0
    #     PANBOT_VISION_CAM=0
    #     PANBOT_POLICY1_DURATION=10
    cfg.setdefault("robot", {})
    cfg.setdefault("vision", {})
    cfg.setdefault("task1", {})
    cfg.setdefault("policy1", {})
    cfg.setdefault("policy2", {})
    cfg.setdefault("timing", {})
    cfg.setdefault("ui", {})
    cfg.setdefault("paths", {})

    cfg["robot"]["port"] = _env_override_str("PANBOT_ROBOT_PORT", cfg["robot"].get("port", "/dev/ttyACM0"))
    cfg["robot"]["id"] = _env_override_str("PANBOT_ROBOT_ID", cfg["robot"].get("id", "my_awesome_follower_arm"))
    cfg["robot"]["calibration_dir"] = _env_override_str(
        "PANBOT_ROBOT_CALIB_DIR", cfg["robot"].get("calibration_dir", "")
    )

    cfg["vision"]["cam_index"] = _env_override_int("PANBOT_VISION_CAM", int(cfg["vision"].get("cam_index", 0)))
    cfg["vision"]["backend"] = _env_override_str("PANBOT_VISION_BACKEND", cfg["vision"].get("backend", "v4l2"))
    cfg["vision"]["mjpg"] = bool(int(_env_override_str("PANBOT_VISION_MJPG", str(int(cfg["vision"].get("mjpg", 1))))))

    cfg["vision"]["width"] = _env_override_int("PANBOT_VISION_W", int(cfg["vision"].get("width", 3840)))
    cfg["vision"]["height"] = _env_override_int("PANBOT_VISION_H", int(cfg["vision"].get("height", 2160)))
    cfg["vision"]["fps"] = _env_override_int("PANBOT_VISION_FPS", int(cfg["vision"].get("fps", 30)))

    cfg["ui"]["show"] = bool(int(_env_override_str("PANBOT_SHOW", str(int(cfg["ui"].get("show", 1))))))
    cfg["ui"]["yolo_preview_scale"] = _env_override_float("PANBOT_YOLO_PREVIEW", float(cfg["ui"].get("yolo_preview_scale", 0.55)))
    cfg["ui"]["gru_preview_scale"] = _env_override_float("PANBOT_GRU_PREVIEW", float(cfg["ui"].get("gru_preview_scale", 0.30)))

    cfg["task1"]["fps"] = _env_override_int("PANBOT_TASK_HZ", int(cfg["task1"].get("fps", 30)))
    cfg["task1"]["ramp_time_s"] = _env_override_float("PANBOT_TASK1_RAMP", float(cfg["task1"].get("ramp_time_s", 3.0)))
    cfg["task1"]["pose_hold_s"] = _env_override_float("PANBOT_TASK1_HOLD", float(cfg["task1"].get("pose_hold_s", 1.0)))

    cfg["timing"]["wait_task2_to_task3_s"] = _env_override_float(
        "PANBOT_WAIT_23", float(cfg["timing"].get("wait_task2_to_task3_s", 30.0))
    )

    cfg["policy1"]["repo_id"] = _env_override_str("PANBOT_POLICY1_REPO", cfg["policy1"].get("repo_id", ""))
    cfg["policy1"]["duration_s"] = _env_override_float("PANBOT_POLICY1_DURATION", float(cfg["policy1"].get("duration_s", 10.0)))

    cfg["policy2"]["repo_id"] = _env_override_str("PANBOT_POLICY2_REPO", cfg["policy2"].get("repo_id", ""))
    cfg["policy2"]["duration_s"] = _env_override_float("PANBOT_POLICY2_DURATION", float(cfg["policy2"].get("duration_s", 10.0)))

    return cfg


def _build_so101_config(robot_cfg: Dict[str, Any]) -> SO101FollowerConfig:
    """
    ✅ policy에서 쓸 카메라들은 SO101FollowerConfig.cameras에 넣어야 함
    (vision 카메라와는 별개)
    """
    port = str(robot_cfg.get("port", "/dev/ttyACM0"))
    rid = str(robot_cfg.get("id", "my_awesome_follower_arm"))
    calib_dir = str(robot_cfg.get("calibration_dir", "")).strip() or None

    cameras = robot_cfg.get("cameras", {}) or {}

    cfg = SO101FollowerConfig(port=port, id=rid, cameras=cameras)
    if calib_dir is not None:
        cfg.calibration_dir = calib_dir
    return cfg


def _best_effort_safe_pose(robot, base_pose_ctrl: BasePoseController, seconds: float = 1.0) -> None:
    """
    어떤 에러가 나도 마지막에 base pose로 보내도록 시도
    """
    try:
        base_pose_ctrl.enable()
        end = time.perf_counter() + max(0.2, float(seconds))
        while time.perf_counter() < end:
            base_pose_ctrl.tick()
            time.sleep(0.01)
        logging.info("[SAFE] best-effort base pose sent")
    except Exception as e:
        logging.error("[SAFE] failed: %s", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="Panbot/config/runtime.yaml")
    args = ap.parse_args()

    yaml_path = Path(args.config).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"runtime.yaml not found: {yaml_path}")

    cfg = _read_runtime_config(yaml_path)

    # logging
    log_dir = Path(cfg.get("log_dir", "Panbot/logs")).expanduser().resolve()
    _setup_logging(log_dir)
    logging.info("[CFG] loaded: %s", yaml_path)
    logging.info("[CFG] %s", cfg)

    # paths
    paths = cfg.get("paths", {}) or {}
    corners = Path(paths.get("corners", "Panbot/vision/calibration/corners.json")).expanduser().resolve()
    yolo_model = Path(paths.get("yolo_model", "")).expanduser().resolve()
    gru_ckpt = Path(paths.get("gru_ckpt", "")).expanduser().resolve()

    _require_file(corners, "corners.json")
    _require_file(yolo_model, "yolo_model")
    _require_file(gru_ckpt, "gru_ckpt")

    rp = RuntimePaths(corners=corners, yolo_model=yolo_model, gru_ckpt=gru_ckpt)

    # vision cam config
    vcfg = cfg.get("vision", {}) or {}
    cam_index = int(vcfg.get("cam_index", 0))
    backend = str(vcfg.get("backend", "v4l2"))
    mjpg = bool(vcfg.get("mjpg", True))
    width = int(vcfg.get("width", 3840))
    height = int(vcfg.get("height", 2160))
    fps = int(vcfg.get("fps", 30))

    # ui
    uicfg = cfg.get("ui", {}) or {}
    show = bool(uicfg.get("show", True))
    yolo_preview_scale = float(uicfg.get("yolo_preview_scale", 0.55))
    gru_preview_scale = float(uicfg.get("gru_preview_scale", 0.30))

    # robot
    robot_cfg_dict = cfg.get("robot", {}) or {}
    robot_cfg = _build_so101_config(robot_cfg_dict)
    robot = make_robot_from_config(robot_cfg)

    # Base pose controller config
    base_pose = cfg.get("base_pose", None) or DEFAULT_BASE_POSE
    hold_cfg = HoldConfig(
        fps=int(cfg.get("task1", {}).get("fps", 30)),
        hold_interval_s=float(cfg.get("base_pose_hold_interval_s", 0.25)),
        use_current_for_missing=True,
    )

    stop_flag = {"stop": False}

    def _sig_handler(signum, frame):
        stop_flag["stop"] = True
        logging.info("[SIGNAL] received signum=%s -> stopping...", signum)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    cap = None
    base_ctrl = None

    # windows
    yolo_win = "YOLO"
    gru_win = "GRU"

    try:
        # 1) connect robot
        logging.info("[ROBOT] connect...")
        robot.connect()
        action_features = set(robot.action_features.keys())

        base_ctrl = BasePoseController(robot, hold_cfg, action_features=action_features)
        base_ctrl.set_target(base_pose)
        base_ctrl.enable()

        # 2) open vision camera
        logging.info("[VISION] open camera...")
        cap = open_camera(
            cam_index=cam_index,
            backend=backend,
            mjpg=mjpg,
            width=width,
            height=height,
            fps=fps,
        )
        if not cap.isOpened():
            raise RuntimeError(f"Vision camera open failed: index={cam_index}, backend={backend}")

        last_frame_ok_t = time.perf_counter()
        watchdog_s = float(cfg.get("vision_watchdog_s", 2.0))

        # 3) build vision infer objects (re-use your modules)
        yolo_cfg = YOLOSegConfig(
            model_path=rp.yolo_model,
            use_warp=True,
            corners_path=rp.corners,
            warp_w=int(cfg.get("yolo", {}).get("warp_w", 0)),
            warp_h=int(cfg.get("yolo", {}).get("warp_h", 0)),
            area_thr_ratio=float(cfg.get("yolo", {}).get("area_thr_ratio", 0.17)),
            hold_frames=int(cfg.get("yolo", {}).get("hold_frames", 30)),
            conf=float(cfg.get("yolo", {}).get("conf", 0.25)),
            imgsz=int(cfg.get("yolo", {}).get("imgsz", 640)),
        )
        yolo = YOLOSegInfer(yolo_cfg)

        gru_cfg = GRUInferConfig(
            checkpoint_path=rp.gru_ckpt,
            use_warp=True,
            corners_path=rp.corners,
            warp_w=int(cfg.get("gru", {}).get("warp_w", 0)),
            warp_h=int(cfg.get("gru", {}).get("warp_h", 0)),
            image_size=int(cfg.get("gru", {}).get("image_size", 224)),
            seq_len=int(cfg.get("gru", {}).get("seq_len", 16)),
            stride=int(cfg.get("gru", {}).get("stride", 6)),
            ema=float(cfg.get("gru", {}).get("ema", 0.7)),
            ready_hold=int(cfg.get("gru", {}).get("ready_hold", 3)),
            amp=bool(cfg.get("gru", {}).get("amp", True)),
        )
        gru = GRUInfer(gru_cfg)

        # 4) Task1 stepper
        t1cfg = Task1MotionConfig(
            fps=int(cfg.get("task1", {}).get("fps", 30)),
            ramp_time_s=float(cfg.get("task1", {}).get("ramp_time_s", 3.0)),
            pose_hold_s=float(cfg.get("task1", {}).get("pose_hold_s", 1.0)),
        )
        task1 = Task1MotionStepper(robot, t1cfg, action_features=action_features)

        # =============================
        # STAGE A: Task1 + YOLO trigger (interrupt)
        # =============================
        logging.info("[STAGE1] start Task1 INITIAL + YOLO trigger")
        base_ctrl.disable()          # task1이 로봇 제어권 가짐
        task1.start_initial()

        stage = "TASK1"
        yolo_triggered = False

        # fixed loop rate
        main_hz = int(cfg.get("main_hz", 30))
        dt_main = 1.0 / max(1, main_hz)

        while not stop_flag["stop"]:
            loop_start = time.perf_counter()

            ok, frame = cap.read()
            if ok and frame is not None:
                last_frame_ok_t = time.perf_counter()
            else:
                # watchdog
                if (time.perf_counter() - last_frame_ok_t) > watchdog_s:
                    raise RuntimeError("[FAILSAFE] Vision camera watchdog timeout")
                time.sleep(0.001)
                continue

            if stage == "TASK1":
                trig, vis, info = yolo.step(frame)
                if trig and not yolo_triggered:
                    yolo_triggered = True
                    logging.info("[YOLO] TRIGGER ✅ info=%s", info)
                    task1.interrupt_to_return()

                # task1 tick (30Hz stepper)
                now = time.perf_counter()
                task1.step(now)

                if show:
                    cv2.imshow(yolo_win, resize_for_preview(vis, yolo_preview_scale))
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27):
                        stop_flag["stop"] = True

                # return이 끝나면 Stage2로
                if task1.is_return_done():
                    logging.info("[STAGE1] Task1 RETURN done -> Stage2(GRU wait)")
                    stage = "WAIT_GRU"
                    base_ctrl.enable()
                    gru.reset()
                    # yolo window 닫고 싶으면 닫기
                    if show:
                        try:
                            cv2.destroyWindow(yolo_win)
                        except Exception:
                            pass

            elif stage == "WAIT_GRU":
                # 기본자세 유지
                base_ctrl.tick()

                trig, vis, info = gru.step(frame)
                if show:
                    cv2.imshow(gru_win, resize_for_preview(vis, gru_preview_scale))
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27):
                        stop_flag["stop"] = True

                if trig:
                    logging.info("[GRU] TRIGGER ✅ info=%s", info)
                    stage = "RUN_POLICY1"
                    break

            # loop sleep
            elapsed = time.perf_counter() - loop_start
            to_sleep = dt_main - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

        if stop_flag["stop"]:
            logging.info("[STOP] requested before policies.")
            return

        # 이제 vision은 더 이상 필요 없음(요구사항 기준)
        if show:
            try:
                cv2.destroyWindow(gru_win)
            except Exception:
                pass
        cap.release()
        cap = None

        # =============================
        # STAGE B: Policy1
        # =============================
        p1 = cfg.get("policy1", {}) or {}
        repo1 = str(p1.get("repo_id", "")).strip()
        if not repo1:
            raise ValueError("policy1.repo_id is empty in runtime.yaml")
        dur1 = float(p1.get("duration_s", 10.0))

        logging.info("[STAGE3] run policy1 repo=%s duration=%.1fs", repo1, dur1)
        base_ctrl.disable()
        run_pretrained_policy_shared_robot(
            robot=robot,
            repo_id=repo1,
            fps=int(p1.get("fps", 30)),
            duration_s=dur1,
            task=p1.get("task", None),
            rename_map=p1.get("rename_map", None),
            dataset_repo_id=p1.get("dataset_repo_id", None),
            dataset_root=p1.get("dataset_root", None),
            use_amp=bool(p1.get("use_amp", True)),
            print_joints=bool(p1.get("print_joints", False)),
            print_joints_every=int(p1.get("print_joints_every", 30)),
        )

        # policy 끝나면 기본자세
        base_ctrl.enable()
        _best_effort_safe_pose(robot, base_ctrl, seconds=1.0)

        # wait
        wait_s = float(cfg.get("timing", {}).get("wait_task2_to_task3_s", 30.0))
        logging.info("[WAIT] %.1fs at base pose...", wait_s)
        t_end = time.perf_counter() + wait_s
        while time.perf_counter() < t_end and not stop_flag["stop"]:
            loop_start = time.perf_counter()
            base_ctrl.tick()
            elapsed = time.perf_counter() - loop_start
            to_sleep = dt_main - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

        if stop_flag["stop"]:
            logging.info("[STOP] requested before policy2.")
            return

        # =============================
        # STAGE C: Policy2
        # =============================
        p2 = cfg.get("policy2", {}) or {}
        repo2 = str(p2.get("repo_id", "")).strip()
        if not repo2:
            raise ValueError("policy2.repo_id is empty in runtime.yaml")
        dur2 = float(p2.get("duration_s", 10.0))

        logging.info("[STAGE4] run policy2 repo=%s duration=%.1fs", repo2, dur2)
        base_ctrl.disable()
        run_pretrained_policy_shared_robot(
            robot=robot,
            repo_id=repo2,
            fps=int(p2.get("fps", 30)),
            duration_s=dur2,
            task=p2.get("task", None),
            rename_map=p2.get("rename_map", None),
            dataset_repo_id=p2.get("dataset_repo_id", None),
            dataset_root=p2.get("dataset_root", None),
            use_amp=bool(p2.get("use_amp", True)),
            print_joints=bool(p2.get("print_joints", False)),
            print_joints_every=int(p2.get("print_joints_every", 30)),
        )

        base_ctrl.enable()
        _best_effort_safe_pose(robot, base_ctrl, seconds=1.0)

        logging.info("[DONE] main_runtime finished OK ✅")

    except Exception as e:
        logging.exception("[FATAL] %s", e)
        # fail-safe: base pose best-effort
        if robot is not None and base_ctrl is not None:
            _best_effort_safe_pose(robot, base_ctrl, seconds=1.0)
        raise

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if show:
                cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
