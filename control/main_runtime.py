# Panbot/control/main_runtime.py
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import threading
import select
import tty
import termios
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

from Panbot.tasks.base_pose import BasePoseController, HoldConfig, normalize_action_keys
from Panbot.tasks.task1_motion import Task1MotionConfig, Task1MotionStepper, DEFAULT_REST_ACTION
from Panbot.policies.common_policy_runner import run_pretrained_policy_shared_robot


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_POSE = dict(DEFAULT_REST_ACTION)


@dataclass
class RuntimePaths:
    corners: Path
    yolo_model: Path
    gru_ckpt: Path


# -----------------------------
# Key watcher: ESC = stop, q = preview off
# -----------------------------
class KeyWatcher:
    """
    터미널 stdin에서 키를 비동기로 읽어서 이벤트를 세팅합니다.
    - ESC 단독 입력이면 stop_event.set()
    - q 입력이면 preview_off_event.set()  (프리뷰만 끄기)
    """
    def __init__(self, stop_event: threading.Event, preview_off_event: threading.Event):
        self.stop_event = stop_event
        self.preview_off_event = preview_off_event
        self._running = False
        self._t: Optional[threading.Thread] = None
        self._fd = sys.stdin.fileno()
        self._old = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            logging.warning("[KEY] stdin is not a TTY; KeyWatcher disabled.")
            return
        self._running = True
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # 한 글자씩 바로 읽기
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        logging.info("[KEY] KeyWatcher started (ESC=stop, q=preview off)")

    def stop(self) -> None:
        self._running = False
        try:
            if self._old is not None:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        except Exception:
            pass
        logging.info("[KEY] KeyWatcher stopped")

    def _read_byte(self, timeout_s: float) -> Optional[bytes]:
        r, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if r:
            return os.read(self._fd, 1)
        return None

    def _loop(self) -> None:
        while self._running and (not self.stop_event.is_set()):
            b = self._read_byte(0.05)
            if not b:
                continue

            # q → preview off
            if b in (b"q", b"Q"):
                self.preview_off_event.set()
                continue

            # ESC(0x1b) → stop (화살표키 시퀀스는 무시)
            if b == b"\x1b":
                b2 = self._read_byte(0.02)
                if b2 is None:
                    self.stop_event.set()
                # ESC 시퀀스(예: 방향키)면 무시


# -----------------------------
# Utilities
# -----------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime.yaml must be a mapping(dict) at top-level")
    return data


def _require_file(p: Path, name: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"missing {name}: {p}")


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
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(lvl)

    # 중복 방지: 기존 Stream/File 제거
    for h in list(root.handlers):
        if isinstance(h, (logging.StreamHandler, logging.FileHandler)):
            root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setLevel(lvl)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)

    logging.info("==================================================")
    logging.info("[main_runtime] logging started")
    logging.info("log_file=%s", log_path)
    logging.info("==================================================")


def _as_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _dict_get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def _normalize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    runtime.yaml 구조 보장 + 기본값 채우기
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

    # log
    cfg["log"].setdefault("dir", "Panbot/logs")
    cfg["log"].setdefault("level", "INFO")

    # paths
    cfg["paths"].setdefault("corners", "Panbot/vision/calibration/corners.json")
    cfg["paths"].setdefault("yolo_model", "")
    cfg["paths"].setdefault("gru_ckpt", "")

    # robot
    cfg["robot"].setdefault("type", "so101_follower")
    cfg["robot"].setdefault("port", "/dev/ttyACM0")
    cfg["robot"].setdefault("id", "my_awesome_follower_arm")
    cfg["robot"].setdefault("calibration_dir", "")
    cfg["robot"].setdefault("cameras", {})

    # vision
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

    # yolo_trigger
    cfg["yolo_trigger"].setdefault("conf", 0.25)
    cfg["yolo_trigger"].setdefault("imgsz", 640)
    cfg["yolo_trigger"].setdefault("area_thr_ratio", 0.17)
    cfg["yolo_trigger"].setdefault("hold_frames", 30)
    cfg["yolo_trigger"].setdefault("use_warp", True)
    cfg["yolo_trigger"].setdefault("warp_w", 0)
    cfg["yolo_trigger"].setdefault("warp_h", 0)

    # gru_trigger
    cfg["gru_trigger"].setdefault("image_size", 224)
    cfg["gru_trigger"].setdefault("seq_len", 16)
    cfg["gru_trigger"].setdefault("stride", 6)
    cfg["gru_trigger"].setdefault("ema", 0.7)
    cfg["gru_trigger"].setdefault("ready_hold", 3)
    cfg["gru_trigger"].setdefault("amp", True)
    cfg["gru_trigger"].setdefault("use_warp", True)
    cfg["gru_trigger"].setdefault("warp_w", 0)
    cfg["gru_trigger"].setdefault("warp_h", 0)

    # task
    cfg["task"].setdefault("hz", 30)
    cfg["task"].setdefault("task1_ramp_time_s", 3.0)
    cfg["task"].setdefault("task1_pose_hold_s", 1.0)
    cfg["task"].setdefault("task1_return_to_base_ramp_time_s", 1.0)
    cfg["task"].setdefault("task1_initial_ramp_overrides", {})
    cfg["task"].setdefault("task1_return_ramp_overrides", {})
    cfg["task"].setdefault("base_pose_hold_interval_s", 0.25)
    cfg["task"].setdefault("policy_fps", 30)
    cfg["task"].setdefault("task2_duration_s", 10.0)
    cfg["task"].setdefault("task3_duration_s", 10.0)
    cfg["task"].setdefault("wait_task2_to_task3_s", 30.0)

    # poses
    cfg["poses"].setdefault("base_pose", None)
    cfg["poses"].setdefault("task1_initial_sequence", None)
    cfg["poses"].setdefault("task1_return_sequence", None)

    # policies
    cfg["policies"].setdefault("policy1", {})
    cfg["policies"].setdefault("policy2", {})
    cfg["policies"]["policy1"].setdefault("repo_id", "")
    cfg["policies"]["policy2"].setdefault("repo_id", "")

    return cfg


def _import_opencv_camera_config():
    """
    lerobot 버전/구조에 따라 OpenCV 카메라 config import 경로가 다를 수 있어서
    몇 가지 후보를 best-effort로 시도합니다.
    """
    candidates = [
        "lerobot.cameras.opencv.configuration_opencv:OpenCVCameraConfig",
        "lerobot.cameras.opencv:OpenCVCameraConfig",
        "lerobot.cameras.opencv.config_opencv_camera:OpenCVCameraConfig",
        "lerobot.cameras.opencv_camera:OpenCVCameraConfig",
        "lerobot.cameras.configs:OpenCVCameraConfig",
    ]
    for spec in candidates:
        mod, name = spec.split(":")
        try:
            m = __import__(mod, fromlist=[name])
            return getattr(m, name)
        except Exception:
            continue
    raise ImportError(
        "Cannot import OpenCVCameraConfig from lerobot. "
        "lerobot 카메라 config 경로를 확인해 주세요."
    )


def _build_robot_cameras(cameras_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    ✅ runtime.yaml의 robot.cameras(딕셔너리)를
    lerobot이 기대하는 CameraConfig 객체 딕셔너리로 변환합니다.

    (이 변환이 없으면: AttributeError: 'dict' object has no attribute 'width')
    """
    if not cameras_cfg:
        return {}

    OpenCVCameraConfig = _import_opencv_camera_config()

    out: Dict[str, Any] = {}
    for cam_name, cam in cameras_cfg.items():
        if not isinstance(cam, dict):
            out[cam_name] = cam
            continue

        cam_type = str(cam.get("type", "opencv")).lower()
        if cam_type != "opencv":
            raise ValueError(f"Unsupported camera type for robot.cameras.{cam_name}: {cam_type}")

        out[cam_name] = OpenCVCameraConfig(
            index_or_path=cam.get("index_or_path", 0),
            width=int(cam.get("width", 640)),
            height=int(cam.get("height", 480)),
            fps=int(cam.get("fps", 30)),
            fourcc=str(cam.get("fourcc", "MJPG")),
        )
    return out


def _build_so101_config(robot_cfg: Dict[str, Any]) -> SO101FollowerConfig:
    port = str(robot_cfg.get("port", "/dev/ttyACM0"))
    rid = str(robot_cfg.get("id", "my_awesome_follower_arm"))
    calib_dir = str(robot_cfg.get("calibration_dir", "")).strip() or None

    cameras_cfg = robot_cfg.get("cameras", {}) or {}
    cameras = _build_robot_cameras(cameras_cfg)

    cfg = SO101FollowerConfig(port=port, id=rid, cameras=cameras)
    if calib_dir:
        cfg.calibration_dir = calib_dir
    return cfg


def _ramp_to_pose(
    *,
    robot,
    action_features: set[str],
    target_pose: Dict[str, float],
    duration_s: float = 2.5,
    fps: int = 30,
) -> None:
    """
    현재 관절(pos) -> target_pose 로 선형 보간하며 send_action.
    (ESC 종료 시 안전하게 base_pose로 "천천히" 복귀)
    """
    if duration_s <= 0:
        robot.send_action(normalize_action_keys(target_pose))
        return

    obs = robot.get_observation()
    target = normalize_action_keys(target_pose)

    start: Dict[str, float] = {}
    for k in action_features:
        if k.endswith(".pos") and k in obs:
            start[k] = float(obs[k])
        elif k.endswith(".pos") and (k not in obs):
            # 관측에 없으면 0으로 두기보단 target로 시작
            start[k] = float(target.get(k, 0.0))
        else:
            # non-pos action은 여기서 다루지 않음
            pass

    # target에 없는 키는 start 유지(급격한 변화 방지)
    for k in list(start.keys()):
        if k not in target:
            target[k] = start[k]

    steps = max(1, int(duration_s * max(1, fps)))
    dt = 1.0 / max(1, fps)

    for i in range(steps + 1):
        alpha = i / steps
        act = {}
        for k in start:
            act[k] = (1 - alpha) * start[k] + alpha * target[k]
        robot.send_action(act)
        time.sleep(dt)


def _best_effort_safe_pose(
    *,
    robot,
    base_ctrl: BasePoseController,
    action_features: set[str],
    base_pose: Dict[str, float],
    ramp_s: float = 2.5,
    fps: int = 30,
) -> None:
    """
    종료 시:
    1) 보간으로 base_pose 복귀 시도
    2) base_ctrl로 hold 유지 시도
    """
    try:
        _ramp_to_pose(
            robot=robot,
            action_features=action_features,
            target_pose=base_pose,
            duration_s=ramp_s,
            fps=fps,
        )
    except Exception as e:
        logging.warning("[SAFE] ramp_to_pose failed: %s", e)

    try:
        base_ctrl.set_target(base_pose)
        base_ctrl.enable()
        end = time.perf_counter() + 0.8
        while time.perf_counter() < end:
            base_ctrl.tick()
            time.sleep(0.01)
        logging.info("[SAFE] base pose hold sent")
    except Exception as e:
        logging.error("[SAFE] base pose hold failed: %s", e)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="Panbot/config/runtime.yaml")
    args = ap.parse_args()

    yaml_path = Path(args.config).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"runtime.yaml not found: {yaml_path}")

    raw = _load_yaml(yaml_path)
    cfg = _normalize_cfg(raw)

    # logging
    log_dir = _as_path(cfg["log"]["dir"])
    log_level = str(cfg["log"].get("level", "INFO"))
    _setup_logging(log_dir, log_level)
    logging.info("[CFG] loaded: %s", yaml_path)

    # events (ESC / q)
    stop_event = threading.Event()
    preview_off_event = threading.Event()
    keywatch = KeyWatcher(stop_event, preview_off_event)
    keywatch.start()

    # SIGINT/SIGTERM도 stop_event로 통일
    def _sig_handler(signum, frame):
        logging.info("[SIGNAL] received signum=%s -> stopping...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # paths
    paths = cfg["paths"]
    corners = _as_path(paths["corners"])
    yolo_model = _as_path(paths["yolo_model"])
    gru_ckpt = _as_path(paths["gru_ckpt"])

    _require_file(corners, "corners.json")
    _require_file(yolo_model, "yolo_model")
    _require_file(gru_ckpt, "gru_ckpt")

    rp = RuntimePaths(corners=corners, yolo_model=yolo_model, gru_ckpt=gru_ckpt)

    # vision config
    vcfg = cfg["vision"]
    cam_index = int(vcfg["cam_index"])
    backend = str(vcfg["backend"])
    mjpg = bool(vcfg["mjpg"])
    v_w = int(vcfg["width"])
    v_h = int(vcfg["height"])
    v_fps = int(vcfg["fps"])
    show = bool(vcfg["show"])
    yolo_preview_scale = float(vcfg["yolo_preview_scale"])
    gru_preview_scale = float(vcfg["gru_preview_scale"])
    watchdog_s = float(vcfg["watchdog_s"])

    # task config
    tcfg = cfg["task"]
    main_hz = int(tcfg["hz"])
    dt_main = 1.0 / max(1, main_hz)

    task1_ramp = float(tcfg["task1_ramp_time_s"])
    task1_hold = float(tcfg["task1_pose_hold_s"])
    task1_return_to_base_ramp = float(tcfg["task1_return_to_base_ramp_time_s"])
    base_pose_hold_interval = float(tcfg["base_pose_hold_interval_s"])

    policy_fps = int(tcfg["policy_fps"])
    task2_duration = float(tcfg["task2_duration_s"])
    task3_duration = float(tcfg["task3_duration_s"])
    wait_23 = float(tcfg["wait_task2_to_task3_s"])

    # poses
    poses = cfg["poses"]
    base_pose = poses.get("base_pose", None) or DEFAULT_BASE_POSE
    init_seq = poses.get("task1_initial_sequence", None)
    ret_seq = poses.get("task1_return_sequence", None)

    # robot
    robot_cfg = _build_so101_config(cfg["robot"])
    robot = make_robot_from_config(robot_cfg)

    cap = None
    base_ctrl: Optional[BasePoseController] = None

    yolo_win = "YOLO"
    gru_win = "GRU"

    action_features: set[str] = set()

    try:
        # connect
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

        # open vision camera
        logging.info("[VISION] open camera...")
        cap = open_camera(
            cam_index=cam_index,
            backend=backend,
            mjpg=mjpg,
            width=v_w,
            height=v_h,
            fps=v_fps,
        )
        if not cap.isOpened():
            raise RuntimeError(f"Vision camera open failed: index={cam_index}, backend={backend}")

        last_frame_ok_t = time.perf_counter()

        # build infer objects
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

        # Task1 stepper
        def _parse_ramp_overrides(raw):
            if not isinstance(raw, dict):
                return {}
            parsed = {}
            for k, v in raw.items():
                try:
                    idx = int(k)
                    parsed[idx] = float(v)
                except Exception:
                    continue
            return parsed

        initial_ramp_overrides = _parse_ramp_overrides(tcfg.get("task1_initial_ramp_overrides", {}))
        return_ramp_overrides = _parse_ramp_overrides(tcfg.get("task1_return_ramp_overrides", {}))

        t1cfg = Task1MotionConfig(
            fps=main_hz,
            ramp_time_s=task1_ramp,
            pose_hold_s=task1_hold,
            initial_ramp_overrides=initial_ramp_overrides,
            return_ramp_overrides=return_ramp_overrides,
        )
        # sequences override
        if isinstance(init_seq, list) and len(init_seq) > 0:
            t1cfg.initial_sequence = init_seq
        if isinstance(ret_seq, list) and len(ret_seq) > 0:
            t1cfg.return_sequence = ret_seq

        task1 = Task1MotionStepper(robot, t1cfg, action_features=action_features)

        # -------------------------
        # STAGE 1: Task1 + YOLO trigger
        # -------------------------
        logging.info("[STAGE1] start Task1 INITIAL + YOLO trigger")
        base_ctrl.disable()  # task1이 로봇 제어권 가짐
        task1.start_initial()

        stage = "TASK1"
        yolo_triggered = False

        while not stop_event.is_set():
            loop_start = time.perf_counter()

            # q → preview only off
            if preview_off_event.is_set() and show:
                show = False
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                logging.info("[UI] preview off (q)")

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

                task1.step(time.perf_counter())

                if show:
                    cv2.imshow(yolo_win, resize_for_preview(vis, yolo_preview_scale))
                    k = cv2.waitKey(1) & 0xFF
                    # window에서 ESC도 stop
                    if k == 27:
                        stop_event.set()
                    # window에서 q도 preview off
                    if k in (ord("q"), ord("Q")):
                        preview_off_event.set()

                if task1.is_return_done():
                    logging.info("[STAGE1] Task1 RETURN done -> Stage2(GRU wait)")
                    stage = "WAIT_GRU"
                    base_ctrl.ramp_to_target(
                        duration_s=task1_return_to_base_ramp,
                        target_action=base_pose,
                    )
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
                    if k == 27:
                        stop_event.set()
                    if k in (ord("q"), ord("Q")):
                        preview_off_event.set()

                if trig:
                    logging.info("[GRU] TRIGGER ✅ info=%s", info)
                    break

            # fixed loop rate
            elapsed = time.perf_counter() - loop_start
            to_sleep = dt_main - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

        # ESC/Signal로 빠져나온 경우: 즉시 base_pose 복귀 후 종료
        if stop_event.is_set():
            logging.info("[STOP] requested during vision stage -> safe exit")
            if base_ctrl is not None:
                base_ctrl.enable()
                _best_effort_safe_pose(
                    robot=robot,
                    base_ctrl=base_ctrl,
                    action_features=action_features,
                    base_pose=base_pose,
                    ramp_s=2.5,
                    fps=main_hz,
                )
            return

        # vision 종료
        if show:
            try:
                cv2.destroyWindow(gru_win)
            except Exception:
                pass
        cap.release()
        cap = None

        # -------------------------
        # STAGE 3: Policy1 (task2)
        # -------------------------
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
            stop_event=stop_event,  # ✅ policy 중 ESC 즉시 탈출
        )

        # policy 중 ESC로 끝났으면: base_pose 복귀 후 종료
        if stop_event.is_set():
            logging.info("[STOP] requested during policy1 -> safe exit")
            base_ctrl.enable()
            _best_effort_safe_pose(
                robot=robot,
                base_ctrl=base_ctrl,
                action_features=action_features,
                base_pose=base_pose,
                ramp_s=2.5,
                fps=policy_fps,
            )
            return

        base_ctrl.enable()
        _best_effort_safe_pose(
            robot=robot,
            base_ctrl=base_ctrl,
            action_features=action_features,
            base_pose=base_pose,
            ramp_s=1.0,
            fps=policy_fps,
        )

        # -------------------------
        # WAIT between policies
        # -------------------------
        logging.info("[WAIT] %.1fs at base pose...", wait_23)
        t_end = time.perf_counter() + wait_23
        while time.perf_counter() < t_end and (not stop_event.is_set()):
            loop_start = time.perf_counter()
            base_ctrl.tick()
            elapsed = time.perf_counter() - loop_start
            to_sleep = dt_main - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

        if stop_event.is_set():
            logging.info("[STOP] requested during wait -> safe exit")
            base_ctrl.enable()
            _best_effort_safe_pose(
                robot=robot,
                base_ctrl=base_ctrl,
                action_features=action_features,
                base_pose=base_pose,
                ramp_s=2.5,
                fps=main_hz,
            )
            return

        # -------------------------
        # STAGE 4: Policy2 (task3)
        # -------------------------
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
            stop_event=stop_event,  # ✅ policy 중 ESC 즉시 탈출
        )

        # policy2 중 ESC
        if stop_event.is_set():
            logging.info("[STOP] requested during policy2 -> safe exit")
            base_ctrl.enable()
            _best_effort_safe_pose(
                robot=robot,
                base_ctrl=base_ctrl,
                action_features=action_features,
                base_pose=base_pose,
                ramp_s=2.5,
                fps=policy_fps,
            )
            return

        base_ctrl.enable()
        _best_effort_safe_pose(
            robot=robot,
            base_ctrl=base_ctrl,
            action_features=action_features,
            base_pose=base_pose,
            ramp_s=1.0,
            fps=policy_fps,
        )

        logging.info("[DONE] main_runtime finished OK ✅")

    except Exception as e:
        logging.exception("[FATAL] %s", e)
        # fail-safe
        try:
            if base_ctrl is not None and action_features:
                _best_effort_safe_pose(
                    robot=robot,
                    base_ctrl=base_ctrl,
                    action_features=action_features,
                    base_pose=base_pose,
                    ramp_s=2.5,
                    fps=max(main_hz, 1),
                )
        except Exception:
            pass
        raise

    finally:
        try:
            keywatch.stop()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
