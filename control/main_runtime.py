# Panbot/control/main_runtime.py
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import threading
import termios
import tty
import select
import termios

from dataclasses import dataclass, is_dataclass, fields
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
DEFAULT_BASE_POSE = dict(DEFAULT_REST_ACTION)


@dataclass
class RuntimePaths:
    corners: Path
    yolo_model: Path
    gru_ckpt: Path


# -----------------------------
# Env override helpers
# -----------------------------
def _env_get(key: str) -> Optional[str]:
    v = os.environ.get(key, None)
    if v is None:
        return None
    v = str(v).strip()
    return v if v != "" else None


def _env_override_str(key: str, current: Optional[str]) -> Optional[str]:
    v = _env_get(key)
    return v if v is not None else current


def _env_override_int(key: str, current: int) -> int:
    v = _env_get(key)
    return int(v) if v is not None else int(current)


def _env_override_float(key: str, current: float) -> float:
    v = _env_get(key)
    return float(v) if v is not None else float(current)


def _env_override_bool(key: str, current: bool) -> bool:
    """
    허용: 1/0, true/false, yes/no, on/off
    """
    v = _env_get(key)
    if v is None:
        return bool(current)
    s = v.lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    # 숫자로도 처리
    try:
        return bool(int(s))
    except Exception:
        raise ValueError(f"Invalid bool env: {key}={v}")


# -----------------------------
# Logging
# -----------------------------
def _setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """
    - 터미널 + 파일 동시 출력
    - 중복 핸들러 방지
    """
    init_logging()
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"main_runtime_{ts}.log"

    root = logging.getLogger()

    # 레벨
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(lvl)

    # 기존 핸들러 정리(중복 방지): Stream/File만 제거
    for h in list(root.handlers):
        if isinstance(h, (logging.StreamHandler, logging.FileHandler)):
            root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setLevel(lvl)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(lvl)
    stream_handler.setFormatter(fmt)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    logging.info("==================================================")
    logging.info("[main_runtime] logging started")
    logging.info("log_file=%s", log_path)
    logging.info("==================================================")


def _start_esc_listener(stop_flag: dict) -> Optional[threading.Thread]:
    """
    터미널(STDIN)에서 ESC(27) 입력을 감지해서 stop_flag["stop"]=True로 만듭니다.
    (cv2 창이 없어도 policy 중에 동작)
    """
    if not sys.stdin.isatty():
        logging.warning("[KEY] stdin is not a TTY -> ESC listener disabled")
        return None

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    def _worker():
        try:
            tty.setcbreak(fd)  # 한 글자씩 즉시 읽기
            while not stop_flag["stop"]:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch and ord(ch) == 27:  # ESC
                    stop_flag["stop"] = True
                    logging.info("[KEY] ESC pressed -> stopping...")
                    break
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    return th


# -----------------------------
# YAML
# -----------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime.yaml must be a mapping(dict) at top-level")
    return data


def _require_file(p: Path, name: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"missing {name}: {p}")


def _normalize_runtime_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    ✅ runtime.yaml 구조에 맞게 섹션 보장 + env override 적용

    Env override (필요한 것만):
      - PANBOT_ROBOT_PORT
      - PANBOT_ROBOT_ID
      - PANBOT_ROBOT_CALIB_DIR

      - PANBOT_VISION_CAM
      - PANBOT_VISION_BACKEND
      - PANBOT_VISION_MJPG
      - PANBOT_VISION_W / H / FPS
      - PANBOT_SHOW
      - PANBOT_YOLO_PREVIEW
      - PANBOT_GRU_PREVIEW
      - PANBOT_VISION_WATCHDOG

      - PANBOT_TASK_HZ
      - PANBOT_TASK1_RAMP
      - PANBOT_TASK1_HOLD
      - PANBOT_BASE_HOLD
      - PANBOT_POLICY_FPS
      - PANBOT_TASK2_DURATION
      - PANBOT_TASK3_DURATION
      - PANBOT_WAIT_23
    """
    cfg.setdefault("paths", {})
    cfg.setdefault("log", {})
    cfg.setdefault("robot", {})
    cfg.setdefault("vision", {})
    cfg.setdefault("yolo_trigger", {})
    cfg.setdefault("gru_trigger", {})
    cfg.setdefault("task", {})
    cfg.setdefault("poses", {})
    cfg.setdefault("policies", {})

    # ---- log
    cfg["log"].setdefault("dir", "Panbot/logs")
    cfg["log"].setdefault("level", "INFO")

    # ---- robot
    cfg["robot"].setdefault("type", "so101_follower")
    cfg["robot"].setdefault("port", "/dev/ttyACM0")
    cfg["robot"].setdefault("id", "my_awesome_follower_arm")
    cfg["robot"].setdefault("calibration_dir", "")
    cfg["robot"].setdefault("cameras", {})

    cfg["robot"]["port"] = _env_override_str("PANBOT_ROBOT_PORT", str(cfg["robot"]["port"]))
    cfg["robot"]["id"] = _env_override_str("PANBOT_ROBOT_ID", str(cfg["robot"]["id"]))
    cfg["robot"]["calibration_dir"] = _env_override_str(
        "PANBOT_ROBOT_CALIB_DIR", str(cfg["robot"].get("calibration_dir", ""))
    )

    # ---- vision
    cfg["vision"].setdefault("cam_index", 0)
    cfg["vision"].setdefault("backend", "v4l2")
    cfg["vision"].setdefault("mjpg", True)
    cfg["vision"].setdefault("width", 3840)
    cfg["vision"].setdefault("height", 2160)
    cfg["vision"].setdefault("fps", 30)
    cfg["vision"].setdefault("show", True)
    cfg["vision"].setdefault("yolo_preview_scale", 0.55)
    cfg["vision"].setdefault("gru_preview_scale", 0.30)
    cfg["vision"].setdefault("watchdog_s", 2.0)

    cfg["vision"]["cam_index"] = _env_override_int("PANBOT_VISION_CAM", int(cfg["vision"]["cam_index"]))
    cfg["vision"]["backend"] = _env_override_str("PANBOT_VISION_BACKEND", str(cfg["vision"]["backend"]))
    cfg["vision"]["mjpg"] = _env_override_bool("PANBOT_VISION_MJPG", bool(cfg["vision"]["mjpg"]))

    cfg["vision"]["width"] = _env_override_int("PANBOT_VISION_W", int(cfg["vision"]["width"]))
    cfg["vision"]["height"] = _env_override_int("PANBOT_VISION_H", int(cfg["vision"]["height"]))
    cfg["vision"]["fps"] = _env_override_int("PANBOT_VISION_FPS", int(cfg["vision"]["fps"]))

    cfg["vision"]["show"] = _env_override_bool("PANBOT_SHOW", bool(cfg["vision"]["show"]))
    cfg["vision"]["yolo_preview_scale"] = _env_override_float(
        "PANBOT_YOLO_PREVIEW", float(cfg["vision"]["yolo_preview_scale"])
    )
    cfg["vision"]["gru_preview_scale"] = _env_override_float(
        "PANBOT_GRU_PREVIEW", float(cfg["vision"]["gru_preview_scale"])
    )
    cfg["vision"]["watchdog_s"] = _env_override_float("PANBOT_VISION_WATCHDOG", float(cfg["vision"]["watchdog_s"]))

    # ---- yolo_trigger
    cfg["yolo_trigger"].setdefault("conf", 0.25)
    cfg["yolo_trigger"].setdefault("imgsz", 640)
    cfg["yolo_trigger"].setdefault("area_thr_ratio", 0.17)
    cfg["yolo_trigger"].setdefault("hold_frames", 30)
    cfg["yolo_trigger"].setdefault("use_warp", True)
    cfg["yolo_trigger"].setdefault("warp_w", 0)
    cfg["yolo_trigger"].setdefault("warp_h", 0)

    # ---- gru_trigger
    cfg["gru_trigger"].setdefault("image_size", 224)
    cfg["gru_trigger"].setdefault("seq_len", 16)
    cfg["gru_trigger"].setdefault("stride", 6)
    cfg["gru_trigger"].setdefault("ema", 0.7)
    cfg["gru_trigger"].setdefault("ready_hold", 3)
    cfg["gru_trigger"].setdefault("amp", True)
    cfg["gru_trigger"].setdefault("use_warp", True)
    cfg["gru_trigger"].setdefault("warp_w", 0)
    cfg["gru_trigger"].setdefault("warp_h", 0)

    # ---- task
    cfg["task"].setdefault("hz", 30)
    cfg["task"].setdefault("task1_ramp_time_s", 3.0)
    cfg["task"].setdefault("task1_pose_hold_s", 1.0)
    cfg["task"].setdefault("base_pose_hold_interval_s", 0.25)
    cfg["task"].setdefault("policy_fps", 30)
    cfg["task"].setdefault("task2_duration_s", 10.0)
    cfg["task"].setdefault("task3_duration_s", 10.0)
    cfg["task"].setdefault("wait_task2_to_task3_s", 30.0)

    cfg["task"]["hz"] = _env_override_int("PANBOT_TASK_HZ", int(cfg["task"]["hz"]))
    cfg["task"]["task1_ramp_time_s"] = _env_override_float(
        "PANBOT_TASK1_RAMP", float(cfg["task"]["task1_ramp_time_s"])
    )
    cfg["task"]["task1_pose_hold_s"] = _env_override_float(
        "PANBOT_TASK1_HOLD", float(cfg["task"]["task1_pose_hold_s"])
    )
    cfg["task"]["base_pose_hold_interval_s"] = _env_override_float(
        "PANBOT_BASE_HOLD", float(cfg["task"]["base_pose_hold_interval_s"])
    )
    cfg["task"]["policy_fps"] = _env_override_int("PANBOT_POLICY_FPS", int(cfg["task"]["policy_fps"]))
    cfg["task"]["task2_duration_s"] = _env_override_float(
        "PANBOT_TASK2_DURATION", float(cfg["task"]["task2_duration_s"])
    )
    cfg["task"]["task3_duration_s"] = _env_override_float(
        "PANBOT_TASK3_DURATION", float(cfg["task"]["task3_duration_s"])
    )
    cfg["task"]["wait_task2_to_task3_s"] = _env_override_float(
        "PANBOT_WAIT_23", float(cfg["task"]["wait_task2_to_task3_s"])
    )

    # ---- poses
    cfg["poses"].setdefault("base_pose", None)
    cfg["poses"].setdefault("task1_initial_sequence", None)
    cfg["poses"].setdefault("task1_return_sequence", None)

    # ---- policies
    cfg["policies"].setdefault("policy1", {})
    cfg["policies"].setdefault("policy2", {})
    cfg["policies"]["policy1"].setdefault("repo_id", "")
    cfg["policies"]["policy2"].setdefault("repo_id", "")

    # 선택 env override (repo_id)
    cfg["policies"]["policy1"]["repo_id"] = _env_override_str(
        "PANBOT_POLICY1_REPO", str(cfg["policies"]["policy1"].get("repo_id", ""))
    )
    cfg["policies"]["policy2"]["repo_id"] = _env_override_str(
        "PANBOT_POLICY2_REPO", str(cfg["policies"]["policy2"].get("repo_id", ""))
    )

    return cfg


def _import_opencv_camera_config_cls():
    """
    LeRobot 버전에 따라 OpenCV 카메라 config 클래스 위치가 다를 수 있어서
    여러 후보를 순서대로 시도합니다.
    """
    candidates = [
        ("lerobot.cameras.opencv_camera", "OpenCVCameraConfig"),
        ("lerobot.cameras.opencv", "OpenCVCameraConfig"),
        ("lerobot.cameras.opencv.config", "OpenCVCameraConfig"),
    ]
    last_err = None
    for mod, name in candidates:
        try:
            m = __import__(mod, fromlist=[name])
            return getattr(m, name)
        except Exception as e:
            last_err = e
    raise ImportError(
        "Cannot import OpenCVCameraConfig from lerobot. "
        "Your lerobot version may use a different module path."
    ) from last_err


def _make_dataclass_instance(cls, raw: dict):
    """raw dict에서 dataclass field에 해당하는 키만 골라서 생성"""
    if not is_dataclass(cls):
        return cls(**raw)
    allowed = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in raw.items() if k in allowed}
    return cls(**kwargs)


def _build_robot_camera_configs(cameras_raw: dict) -> dict:
    """
    runtime.yaml의 robot.cameras (dict)를
    LeRobot이 기대하는 CameraConfig 객체 dict로 변환.
    """
    if not cameras_raw:
        return {}

    opencv_cls = _import_opencv_camera_config_cls()

    cameras_out = {}
    for name, cam in cameras_raw.items():
        if cam is None:
            continue
        if not isinstance(cam, dict):
            # 이미 객체면 그대로
            cameras_out[name] = cam
            continue

        cam_type = str(cam.get("type", "opencv")).lower()
        if cam_type != "opencv":
            raise ValueError(f"Unsupported camera type '{cam_type}' for robot.cameras.{name}")

        # YAML 키 이름 그대로 유지하되, dataclass field만 골라서 넣음
        cam_cfg = _make_dataclass_instance(opencv_cls, cam)
        cameras_out[name] = cam_cfg

    return cameras_out



def _build_so101_config(robot_cfg: Dict[str, Any]) -> SO101FollowerConfig:
    port = str(robot_cfg.get("port", "/dev/ttyACM0"))
    rid = str(robot_cfg.get("id", "my_awesome_follower_arm"))
    calib_dir = str(robot_cfg.get("calibration_dir", "")).strip() or None

    cameras_raw = robot_cfg.get("cameras", {}) or {}
    cameras = _build_robot_camera_configs(cameras_raw)  # ✅ dict -> CameraConfig 객체들

    cfg = SO101FollowerConfig(port=port, id=rid, cameras=cameras)
    if calib_dir:
        cfg.calibration_dir = calib_dir
    return cfg



def _best_effort_safe_pose(robot, base_pose_ctrl: BasePoseController, seconds: float = 1.0) -> None:
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

    raw = _load_yaml(yaml_path)
    cfg = _normalize_runtime_config(raw)

    # logging
    log_dir = Path(cfg["log"]["dir"]).expanduser().resolve()
    log_level = str(cfg["log"].get("level", "INFO"))
    _setup_logging(log_dir, log_level)
    logging.info("[CFG] loaded: %s", yaml_path)

    # paths
    paths = cfg.get("paths", {}) or {}
    corners = Path(paths.get("corners", "Panbot/vision/calibration/corners.json")).expanduser().resolve()
    yolo_model = Path(paths.get("yolo_model", "")).expanduser().resolve()
    gru_ckpt = Path(paths.get("gru_ckpt", "")).expanduser().resolve()

    _require_file(corners, "corners.json")
    _require_file(yolo_model, "yolo_model")
    _require_file(gru_ckpt, "gru_ckpt")

    rp = RuntimePaths(corners=corners, yolo_model=yolo_model, gru_ckpt=gru_ckpt)

    # vision cam config (✅ runtime.yaml: vision.*)
    vcfg = cfg["vision"]
    cam_index = int(vcfg["cam_index"])
    backend = str(vcfg["backend"])
    mjpg = bool(vcfg["mjpg"])
    width = int(vcfg["width"])
    height = int(vcfg["height"])
    fps = int(vcfg["fps"])

    # vision UI (✅ runtime.yaml: vision.show / scales)
    show = bool(vcfg["show"])
    yolo_preview_scale = float(vcfg["yolo_preview_scale"])
    gru_preview_scale = float(vcfg["gru_preview_scale"])
    watchdog_s = float(vcfg["watchdog_s"])

    # task (✅ runtime.yaml: task.*)
    tcfg = cfg["task"]
    main_hz = int(tcfg["hz"])
    dt_main = 1.0 / max(1, main_hz)

    task1_ramp = float(tcfg["task1_ramp_time_s"])
    task1_hold = float(tcfg["task1_pose_hold_s"])
    base_pose_hold_interval = float(tcfg["base_pose_hold_interval_s"])

    policy_fps = int(tcfg["policy_fps"])
    task2_duration = float(tcfg["task2_duration_s"])
    task3_duration = float(tcfg["task3_duration_s"])
    wait_23 = float(tcfg["wait_task2_to_task3_s"])

    # robot
    robot_cfg_dict = cfg["robot"]
    robot_cfg = _build_so101_config(robot_cfg_dict)
    robot = make_robot_from_config(robot_cfg)

    # poses (✅ runtime.yaml: poses.*)
    poses = cfg.get("poses", {}) or {}
    base_pose = poses.get("base_pose", None) or DEFAULT_BASE_POSE

    init_seq = poses.get("task1_initial_sequence", None)
    ret_seq = poses.get("task1_return_sequence", None)

    stop_flag = {"stop": False}
    _esc_thread = _start_esc_listener(stop_flag)


    def _sig_handler(signum, frame):
        stop_flag["stop"] = True
        logging.info("[SIGNAL] received signum=%s -> stopping...", signum)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    cap = None
    base_ctrl = None

    yolo_win = "YOLO"
    gru_win = "GRU"

    try:
        # 1) connect robot
        logging.info("[ROBOT] connect...")
        robot.connect()
        action_features = set(robot.action_features.keys())

        # base pose controller
        hold_cfg = HoldConfig(
            fps=main_hz,
            hold_interval_s=base_pose_hold_interval,
            use_current_for_missing=True,
        )
        base_ctrl = BasePoseController(robot, hold_cfg, action_features=action_features)
        base_ctrl.set_target(base_pose)
        base_ctrl.enable()

        # 2) open vision camera (✅ runtime.yaml: vision.cam_index/backend/mjpg/width/height/fps)
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

        # 3) vision infer objects (✅ runtime.yaml: yolo_trigger / gru_trigger)
        ycfg = cfg["yolo_trigger"]
        yolo_cfg = YOLOSegConfig(
            model_path=rp.yolo_model,
            conf=float(ycfg["conf"]),
            imgsz=int(ycfg["imgsz"]),
            use_warp=bool(ycfg["use_warp"]),
            corners_path=rp.corners if bool(ycfg["use_warp"]) else None,
            warp_w=int(ycfg.get("warp_w", 0)),
            warp_h=int(ycfg.get("warp_h", 0)),
            area_thr_ratio=float(ycfg["area_thr_ratio"]),
            hold_frames=int(ycfg["hold_frames"]),
        )
        yolo = YOLOSegInfer(yolo_cfg)

        gcfg = cfg["gru_trigger"]
        gru_cfg = GRUInferConfig(
            checkpoint_path=rp.gru_ckpt,
            use_warp=bool(gcfg["use_warp"]),
            corners_path=rp.corners if bool(gcfg["use_warp"]) else None,
            warp_w=int(gcfg.get("warp_w", 0)),
            warp_h=int(gcfg.get("warp_h", 0)),
            image_size=int(gcfg["image_size"]),
            seq_len=int(gcfg["seq_len"]),
            stride=int(gcfg["stride"]),
            ema=float(gcfg["ema"]),
            ready_hold=int(gcfg["ready_hold"]),
            amp=bool(gcfg["amp"]),
        )
        gru = GRUInfer(gru_cfg)

        # 4) Task1 stepper (✅ runtime.yaml: task.* + poses sequences)
        t1cfg = Task1MotionConfig(
            fps=main_hz,
            ramp_time_s=task1_ramp,
            pose_hold_s=task1_hold,
        )
        # sequences override (있으면 적용)
        if isinstance(init_seq, list) and len(init_seq) > 0:
            t1cfg.initial_sequence = init_seq
        if isinstance(ret_seq, list) and len(ret_seq) > 0:
            t1cfg.return_sequence = ret_seq

        task1 = Task1MotionStepper(robot, t1cfg, action_features=action_features)

        # =============================
        # STAGE 1: Task1 INITIAL + YOLO trigger -> interrupt_to_return
        # =============================
        logging.info("[STAGE1] start Task1 INITIAL + YOLO trigger")
        base_ctrl.disable()
        task1.start_initial()

        stage = "TASK1"
        yolo_triggered = False

        while not stop_flag["stop"]:
            loop_start = time.perf_counter()

            ok, frame = cap.read()
            if ok and frame is not None:
                last_frame_ok_t = time.perf_counter()
            else:
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

                # task1 tick
                task1.step(time.perf_counter())

                if show:
                    cv2.imshow(yolo_win, resize_for_preview(vis, yolo_preview_scale))
                    k = cv2.waitKey(1) & 0xFF
                    # ESC: 전체 종료 + base pose 복귀 (나중에 처리)
                    if k == 27:
                        stop_flag["stop"] = True
                        
                    # q: preview만 끄기
                    elif k == ord("q"):
                        show = False
                        try:
                            cv2.destroyWindow(yolo_win)
                        except Exception:
                            pass


                if task1.is_return_done():
                    logging.info("[STAGE1] Task1 RETURN done -> Stage2(GRU wait)")
                    stage = "WAIT_GRU"
                    base_ctrl.enable()
                    gru.reset()
                    if show:
                        try:
                            cv2.destroyWindow(yolo_win)
                        except Exception:
                            pass

            elif stage == "WAIT_GRU":
                base_ctrl.tick()

                trig, vis, info = gru.step(frame)
                if show:
                    cv2.imshow(gru_win, resize_for_preview(vis, gru_preview_scale))
                    k = cv2.waitKey(1) & 0xFF

                    # ESC: 전체 종료 + base pose 복귀 (나중에 처리)
                    if k == 27:
                        stop_flag["stop"] = True

                    # q: preview만 끄기
                    elif k == ord("q"):
                        show = False
                        try:
                            cv2.destroyWindow(gru_win)
                        except Exception:
                            pass


                if trig:
                    logging.info("[GRU] TRIGGER ✅ info=%s", info)
                    break

            # loop sleep
            elapsed = time.perf_counter() - loop_start
            to_sleep = dt_main - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

        if stop_flag["stop"]:
            logging.info("[STOP] requested before policies.")
            return

        # vision 종료
        if show:
            try:
                cv2.destroyWindow(gru_win)
            except Exception:
                pass
        cap.release()
        cap = None

        # =============================
        # STAGE 3: Policy1 (task2)
        # =============================
        pol = cfg["policies"]
        p1 = pol.get("policy1", {}) or {}
        repo1 = str(p1.get("repo_id", "")).strip()
        if not repo1:
            raise ValueError("policies.policy1.repo_id is empty in runtime.yaml")

        logging.info("[STAGE3] run policy1 repo=%s duration=%.1fs", repo1, task2_duration)
        base_ctrl.disable()
        run_pretrained_policy_shared_robot(
            robot=robot,
            repo_id=repo1,
            fps=policy_fps,
            duration_s=task2_duration,
            task=p1.get("task", None),
            rename_map=p1.get("rename_map", None),
            dataset_repo_id=p1.get("dataset_repo_id", None),
            dataset_root=p1.get("dataset_root", None),
            use_amp=bool(p1.get("use_amp", True)),
            print_joints=bool(p1.get("print_joints", False)),
            print_joints_every=int(p1.get("print_joints_every", 30)),
            stop_fn=lambda: stop_flag["stop"],
        )
        
        if stop_flag["stop"]:
            logging.info("[STOP] requested during policy2 -> go base pose and exit")
            base_ctrl.enable()
            _best_effort_safe_pose(robot, base_ctrl, seconds=1.0)
            return

        base_ctrl.enable()
        _best_effort_safe_pose(robot, base_ctrl, seconds=1.0)

        # wait
        logging.info("[WAIT] %.1fs at base pose...", wait_23)
        t_end = time.perf_counter() + wait_23
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
        # STAGE 4: Policy2 (task3)
        # =============================
        p2 = pol.get("policy2", {}) or {}
        repo2 = str(p2.get("repo_id", "")).strip()
        if not repo2:
            raise ValueError("policies.policy2.repo_id is empty in runtime.yaml")

        logging.info("[STAGE4] run policy2 repo=%s duration=%.1fs", repo2, task3_duration)
        base_ctrl.disable()
        run_pretrained_policy_shared_robot(
            robot=robot,
            repo_id=repo2,
            fps=policy_fps,
            duration_s=task3_duration,
            task=p2.get("task", None),
            rename_map=p2.get("rename_map", None),
            dataset_repo_id=p2.get("dataset_repo_id", None),
            dataset_root=p2.get("dataset_root", None),
            use_amp=bool(p2.get("use_amp", True)),
            print_joints=bool(p2.get("print_joints", False)),
            print_joints_every=int(p2.get("print_joints_every", 30)),
            stop_fn=lambda: stop_flag["stop"],
        )

        if stop_flag["stop"]:
            logging.info("[STOP] requested during policy1 -> go base pose and exit")
            base_ctrl.enable()
            _best_effort_safe_pose(robot, base_ctrl, seconds=1.0)
            return

        
        base_ctrl.enable()
        _best_effort_safe_pose(robot, base_ctrl, seconds=1.0)

        logging.info("[DONE] main_runtime finished OK ✅")

    except Exception as e:
        logging.exception("[FATAL] %s", e)
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
