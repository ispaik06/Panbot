# Panbot/control/main_runtime.py

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from ultralytics import YOLO

import yaml

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots import make_robot_from_config
from lerobot.utils.robot_utils import precise_sleep

from Panbot.policies.common_policy_runner import run_pretrained_policy_shared_robot


# -------------------------
# Logging
# -------------------------
def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"main_runtime_{ts}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    # file
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(sh)
    logger.addHandler(fh)

    return log_path


# -------------------------
# YAML load
# -------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime.yaml root must be a mapping")
    return data


def resolve_path(p: str, repo_root: Path) -> Path:
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()


# -------------------------
# Warp utils
# -------------------------
def load_corners(corners_path: Path) -> np.ndarray:
    data = json.loads(corners_path.read_text(encoding="utf-8"))
    pts = data.get("points", None)
    if not pts or len(pts) != 4:
        raise ValueError("corners.json must contain 'points' with 4 entries")
    return np.array([[float(p["x"]), float(p["y"])] for p in pts], dtype=np.float32)


def compute_warp_size(corners: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = corners
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    out_w = int(round(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    out_h = int(round(max(heightA, heightB)))

    out_w = max(out_w, 2)
    out_h = max(out_h, 2)
    return out_w, out_h


def warp_frame(frame_bgr: np.ndarray, corners: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame_bgr, M, (int(out_w), int(out_h)), flags=cv2.INTER_LINEAR)


def resize_for_preview(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0 or abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nw = max(2, int(round(w * scale)))
    nh = max(2, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def open_capture(index: int, backend: str) -> cv2.VideoCapture:
    b = (backend or "auto").lower()
    if b == "auto":
        return cv2.VideoCapture(index)
    if b == "v4l2":
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    if b == "gstreamer":
        return cv2.VideoCapture(index, cv2.CAP_GSTREAMER)
    if b == "ffmpeg":
        return cv2.VideoCapture(index, cv2.CAP_FFMPEG)
    raise ValueError("backend must be one of: auto | v4l2 | gstreamer | ffmpeg")


def try_set_capture(cap: cv2.VideoCapture, width: int, height: int, fps: float, mjpg: bool):
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))


# -------------------------
# Action utils (robot)
# -------------------------
def normalize_action_keys(action: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in action.items():
        kk = k if k.endswith(".pos") else f"{k}.pos"
        out[kk] = float(v)
    return out


def fill_missing_with_current(robot, action_features: set[str], target: Dict[str, float]) -> Dict[str, float]:
    obs = robot.get_observation()
    out = dict(target)
    for k in action_features:
        if k not in out:
            out[k] = float(obs[k])
    unknown = set(out) - action_features
    if unknown:
        raise ValueError(f"Unknown action keys: {sorted(unknown)}")
    return out


def ramp_to_action_interruptible(
    *,
    robot,
    action_features: set[str],
    target_action: Dict[str, float],
    ramp_time_s: float,
    hz: int,
    interrupt_event: Optional[Event] = None,
) -> bool:
    """
    30Hz로 쪼개서 램프 → interrupt가 오면 즉시 중단(True=끝까지 완료, False=중단)
    """
    if interrupt_event is not None and interrupt_event.is_set():
        return False

    if ramp_time_s <= 0:
        robot.send_action(target_action)
        return True

    obs = robot.get_observation()
    start_action = {k: float(obs[k]) for k in action_features}

    dt = 1.0 / max(1, int(hz))
    start_t = time.perf_counter()

    while True:
        if interrupt_event is not None and interrupt_event.is_set():
            return False

        elapsed = time.perf_counter() - start_t
        alpha = min(1.0, elapsed / float(ramp_time_s))
        a = {k: start_action[k] + alpha * (target_action[k] - start_action[k]) for k in action_features}
        robot.send_action(a)

        if alpha >= 1.0:
            break

        precise_sleep(dt)

    return True


def hold_action_interruptible(
    *,
    robot,
    action: Dict[str, float],
    hold_s: float,
    hz: int,
    interrupt_event: Optional[Event] = None,
) -> bool:
    """
    hold 중에도 interrupt가 오면 즉시 중단(True=끝까지 완료, False=중단)
    """
    if hold_s <= 0:
        return True

    dt = 1.0 / max(1, int(hz))
    end = time.perf_counter() + float(hold_s)

    while time.perf_counter() < end:
        if interrupt_event is not None and interrupt_event.is_set():
            return False
        robot.send_action(action)
        precise_sleep(dt)

    return True


def go_base_pose_best_effort(robot, base_pose: Dict[str, float], hz: int, hold_s: float = 1.0):
    try:
        feats = set(robot.action_features.keys())
        base = fill_missing_with_current(robot, feats, normalize_action_keys(base_pose))
        dt = 1.0 / max(1, int(hz))
        end = time.perf_counter() + max(0.2, float(hold_s))
        while time.perf_counter() < end:
            robot.send_action(base)
            precise_sleep(dt)
    except Exception as e:
        logging.error("[SAFE] go_base_pose_best_effort failed: %s", e)


# -------------------------
# Vision threads (YOLO / GRU)
# -------------------------
class YoloTriggerThread:
    def __init__(
        self,
        *,
        cam_index: int,
        backend: str,
        width: int,
        height: int,
        fps: int,
        mjpg: bool,
        model_path: Path,
        corners: np.ndarray,
        warp_out: Tuple[int, int],
        imgsz: int,
        conf: float,
        area_thr_ratio: float,
        hold_frames: int,
        show: bool,
        preview_scale: float,
    ):
        self.cam_index = cam_index
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.mjpg = mjpg
        self.model_path = model_path
        self.corners = corners
        self.warp_out = warp_out
        self.imgsz = imgsz
        self.conf = conf
        self.area_thr_ratio = float(area_thr_ratio)
        self.hold_frames = max(1, int(hold_frames))
        self.show = bool(show)
        self.preview_scale = float(preview_scale)

        self.stop_event = Event()
        self.trigger_event = Event()

        self._thread: Optional[Thread] = None
        self._last_frame_t = time.perf_counter()

    def start(self):
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.show:
            try:
                cv2.destroyWindow("YOLO_TRIGGER")
            except Exception:
                pass

    def watchdog_ok(self, watchdog_s: float = 2.0) -> bool:
        return (time.perf_counter() - self._last_frame_t) <= watchdog_s

    @staticmethod
    def _pick_largest_mask(masks_bool: np.ndarray) -> np.ndarray:
        areas = masks_bool.reshape(masks_bool.shape[0], -1).sum(axis=1)
        return masks_bool[int(np.argmax(areas))]

    def _run(self):
        logging.info("[YOLO] loading model: %s", self.model_path)
        model = YOLO(str(self.model_path))

        cap = open_capture(self.cam_index, self.backend)
        if not cap.isOpened():
            logging.error("[YOLO] camera open failed index=%d backend=%s", self.cam_index, self.backend)
            return

        try_set_capture(cap, self.width, self.height, float(self.fps), self.mjpg)
        out_w, out_h = self.warp_out

        hit = 0
        dt = 1.0 / max(1, int(self.fps))

        logging.info("[YOLO] start cam=%d (%dx%d@%d) warp=%dx%d thr=%.3f hold_frames=%d",
                     self.cam_index, self.width, self.height, self.fps, out_w, out_h,
                     self.area_thr_ratio, self.hold_frames)

        while not self.stop_event.is_set() and not self.trigger_event.is_set():
            loop_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            self._last_frame_t = time.perf_counter()

            frame = warp_frame(frame, self.corners, out_w, out_h)
            H, W = frame.shape[:2]
            frame_area = float(H * W)

            results = model.predict(frame, imgsz=int(self.imgsz), conf=float(self.conf), verbose=False)
            r = results[0]

            ratio = 0.0
            vis = frame

            if r.masks is not None and r.masks.data is not None and len(r.masks.data) > 0:
                masks = r.masks.data.detach().cpu().numpy()
                masks_bool = masks > 0.5
                largest = self._pick_largest_mask(masks_bool)

                if largest.shape[0] != H or largest.shape[1] != W:
                    largest_u8 = (largest.astype(np.uint8) * 255)
                    largest_u8 = cv2.resize(largest_u8, (W, H), interpolation=cv2.INTER_NEAREST)
                    largest = largest_u8 > 0

                mask_area = float(largest.sum())
                ratio = mask_area / frame_area

                if ratio >= self.area_thr_ratio:
                    hit += 1
                else:
                    hit = 0

                if hit >= self.hold_frames:
                    self.trigger_event.set()

                overlay = frame.copy()
                overlay[largest] = (0, 200, 0)
                vis = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

            if self.show:
                text = f"YOLO ratio={ratio:.3f} thr={self.area_thr_ratio:.3f} hit={hit}/{self.hold_frames} TRIG={self.trigger_event.is_set()}"
                cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("YOLO_TRIGGER", resize_for_preview(vis, self.preview_scale))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self.stop_event.set()
                    break

            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(0.0, dt - elapsed))

        cap.release()
        logging.info("[YOLO] stopped (trigger=%s)", self.trigger_event.is_set())


DEFAULT_LABELS = ["not_ready", "almost_ready", "ready"]


class ResNet18GRU(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        backbone = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)
        self.feat_dim = 512
        self.gru = nn.GRU(
            input_size=self.feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.backbone(x).flatten(1)
        feat = feat.reshape(B, T, -1)
        out, _ = self.gru(feat)
        last = out[:, -1, :]
        return self.head(last)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess_frame_bgr(frame_bgr: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device, non_blocking=True)


class GruTriggerThread:
    def __init__(
        self,
        *,
        cam_index: int,
        backend: str,
        width: int,
        height: int,
        fps: int,
        mjpg: bool,
        ckpt_path: Path,
        corners: np.ndarray,
        warp_out: Tuple[int, int],
        preview_scale: float,
        show: bool,
        image_size: int,
        seq_len: int,
        stride: int,
        ema: float,
        ready_hold: int,
        amp: bool,
    ):
        self.cam_index = cam_index
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.mjpg = mjpg
        self.ckpt_path = ckpt_path
        self.corners = corners
        self.warp_out = warp_out

        self.preview_scale = float(preview_scale)
        self.show = bool(show)

        self.image_size = int(image_size)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.ema = float(ema)
        self.ready_hold = max(1, int(ready_hold))
        self.amp = bool(amp)

        self.stop_event = Event()
        self.trigger_event = Event()

        self._thread: Optional[Thread] = None
        self._last_frame_t = time.perf_counter()

    def start(self):
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.show:
            try:
                cv2.destroyWindow("GRU_TRIGGER")
            except Exception:
                pass

    def watchdog_ok(self, watchdog_s: float = 2.0) -> bool:
        return (time.perf_counter() - self._last_frame_t) <= watchdog_s

    def _run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("[GRU] device=%s ckpt=%s", device, self.ckpt_path)

        ckpt = torch.load(str(self.ckpt_path), map_location="cpu")
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            logging.error("[GRU] invalid checkpoint format (expected dict with key 'model')")
            return

        label2id = ckpt.get("label2id", {name: i for i, name in enumerate(DEFAULT_LABELS)})
        id2label = {int(v): k for k, v in label2id.items()}
        num_classes = len(label2id)

        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        hidden = int(ckpt_args.get("hidden", 256))
        gru_layers = int(ckpt_args.get("gru_layers", 1))

        model = ResNet18GRU(hidden_size=hidden, num_layers=gru_layers, num_classes=num_classes)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)
        model.eval()

        ready_id = label2id.get("ready", num_classes - 1)

        cap = open_capture(self.cam_index, self.backend)
        if not cap.isOpened():
            logging.error("[GRU] camera open failed index=%d backend=%s", self.cam_index, self.backend)
            return
        try_set_capture(cap, self.width, self.height, float(self.fps), self.mjpg)

        out_w, out_h = self.warp_out
        need = (self.seq_len - 1) * self.stride + 1
        buf: list[np.ndarray] = []

        ema_prob = None
        ready_streak = 0
        infer_count = 0
        t0 = time.perf_counter()
        dt = 1.0 / max(1, int(self.fps))

        logging.info("[GRU] start cam=%d (%dx%d@%d) warp=%dx%d need=%d ready_hold=%d",
                     self.cam_index, self.width, self.height, self.fps, out_w, out_h, need, self.ready_hold)

        with torch.no_grad():
            while not self.stop_event.is_set() and not self.trigger_event.is_set():
                loop_start = time.perf_counter()

                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                self._last_frame_t = time.perf_counter()

                frame = warp_frame(frame, self.corners, out_w, out_h)
                buf.append(frame)
                if len(buf) > need:
                    buf = buf[-need:]

                pred_label = "warming_up"
                conf = 0.0

                if len(buf) == need:
                    indices = [need - 1 - self.stride * (self.seq_len - 1 - i) for i in range(self.seq_len)]
                    frames = [buf[idx] for idx in indices]
                    xs = [preprocess_frame_bgr(f, self.image_size, device) for f in frames]
                    x = torch.cat(xs, dim=0).unsqueeze(0)  # (1,T,C,H,W)

                    use_amp = (self.amp and device.type == "cuda")
                    if use_amp:
                        with torch.amp.autocast(device_type="cuda", enabled=True):
                            logits = model(x)
                    else:
                        logits = model(x)

                    prob = torch.softmax(logits, dim=1).float().cpu().numpy()[0]

                    if ema_prob is None:
                        ema_prob = prob
                    else:
                        ema_prob = self.ema * ema_prob + (1.0 - self.ema) * prob

                    pred_id = int(np.argmax(ema_prob))
                    conf = float(ema_prob[pred_id])
                    pred_label = id2label.get(pred_id, str(pred_id))

                    if pred_id == ready_id:
                        ready_streak += 1
                    else:
                        ready_streak = 0

                    if ready_streak >= self.ready_hold:
                        self.trigger_event.set()

                    infer_count += 1

                if self.show:
                    fps_est = infer_count / max(1e-6, (time.perf_counter() - t0))
                    vis = frame.copy()
                    text = f"GRU {pred_label} conf={conf:.2f} ready={ready_streak}/{self.ready_hold} TRIG={self.trigger_event.is_set()} fps~{fps_est:.1f}"
                    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow("GRU_TRIGGER", resize_for_preview(vis, self.preview_scale))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self.stop_event.set()
                        break

                elapsed = time.perf_counter() - loop_start
                precise_sleep(max(0.0, dt - elapsed))

        cap.release()
        logging.info("[GRU] stopped (trigger=%s)", self.trigger_event.is_set())


# -------------------------
# Robot config builder (SO101 + cameras for policy)
# -------------------------
def build_so101_config(robot_cfg: Dict[str, Any]) -> SO101FollowerConfig:
    port = str(robot_cfg.get("port", "/dev/ttyACM0"))
    rid = str(robot_cfg.get("id", "my_awesome_follower_arm"))
    calib_dir = str(robot_cfg.get("calibration_dir", "")).strip() or None

    cameras_cfg = robot_cfg.get("cameras", {}) or {}
    cameras: Dict[str, Any] = {}
    for name, c in cameras_cfg.items():
        if not isinstance(c, dict):
            raise ValueError(f"robot.cameras.{name} must be a dict")
        if str(c.get("type", "")).lower() != "opencv":
            raise ValueError("Only opencv cameras are supported in this runtime")
        cameras[name] = OpenCVCameraConfig(
            index_or_path=c.get("index_or_path"),
            width=int(c.get("width", 640)),
            height=int(c.get("height", 480)),
            fps=int(c.get("fps", 30)),
            fourcc=str(c.get("fourcc", "MJPG")),
        )

    cfg = SO101FollowerConfig(port=port, id=rid, cameras=cameras)
    if calib_dir:
        cfg.calibration_dir = calib_dir
    return cfg


# -------------------------
# Task1 (interruptible)
# -------------------------
def run_task1_with_interrupt(
    *,
    robot,
    hz: int,
    base_pose: Dict[str, float],
    initial_sequence: list[Dict[str, float]],
    return_sequence: list[Dict[str, float]],
    ramp_time_s: float,
    pose_hold_s: float,
    interrupt_event: Event,
) -> None:
    feats = set(robot.action_features.keys())
    base = fill_missing_with_current(robot, feats, normalize_action_keys(base_pose))

    init_seq = [fill_missing_with_current(robot, feats, normalize_action_keys(p)) for p in initial_sequence]
    ret_seq = [fill_missing_with_current(robot, feats, normalize_action_keys(p)) for p in return_sequence]

    # 0) base
    logging.info("[TASK1] goto BASE")
    ramp_to_action_interruptible(robot=robot, action_features=feats, target_action=base,
                                ramp_time_s=float(ramp_time_s), hz=int(hz), interrupt_event=None)
    hold_action_interruptible(robot=robot, action=base, hold_s=0.3, hz=int(hz), interrupt_event=None)

    # 1) initial sequence (interruptible)
    logging.info("[TASK1] initial sequence start (%d poses)", len(init_seq))
    last_pose = base
    for i, pose in enumerate(init_seq, start=1):
        logging.info("[TASK1] initial pose %d/%d", i, len(init_seq))
        ok = ramp_to_action_interruptible(
            robot=robot, action_features=feats, target_action=pose,
            ramp_time_s=float(ramp_time_s), hz=int(hz), interrupt_event=interrupt_event
        )
        last_pose = pose
        if not ok:
            logging.info("[TASK1] INTERRUPT during ramp -> jump to RETURN")
            break

        ok2 = hold_action_interruptible(
            robot=robot, action=pose, hold_s=float(pose_hold_s), hz=int(hz), interrupt_event=interrupt_event
        )
        if not ok2:
            logging.info("[TASK1] INTERRUPT during hold -> jump to RETURN")
            break

    # 2) hold until interrupt (if not yet)
    if not interrupt_event.is_set():
        logging.info("[TASK1] holding current pose (waiting YOLO trigger)...")
        while not interrupt_event.is_set():
            robot.send_action(last_pose)
            precise_sleep(1.0 / max(1, int(hz)))

    # 3) return sequence
    logging.info("[TASK1] return sequence start (%d poses)", len(ret_seq))
    for i, pose in enumerate(ret_seq, start=1):
        logging.info("[TASK1] return pose %d/%d", i, len(ret_seq))
        ramp_to_action_interruptible(robot=robot, action_features=feats, target_action=pose,
                                    ramp_time_s=float(ramp_time_s), hz=int(hz), interrupt_event=None)
        hold_action_interruptible(robot=robot, action=pose, hold_s=float(pose_hold_s), hz=int(hz), interrupt_event=None)

    # 4) back to base
    logging.info("[TASK1] back to BASE")
    ramp_to_action_interruptible(robot=robot, action_features=feats, target_action=base,
                                ramp_time_s=float(ramp_time_s), hz=int(hz), interrupt_event=None)
    hold_action_interruptible(robot=robot, action=base, hold_s=0.5, hz=int(hz), interrupt_event=None)


def hold_base_pose_until(robot, base_pose: Dict[str, float], hz: int, until_event: Event) -> None:
    feats = set(robot.action_features.keys())
    base = fill_missing_with_current(robot, feats, normalize_action_keys(base_pose))
    dt = 1.0 / max(1, int(hz))
    logging.info("[IDLE] holding BASE until trigger...")
    while not until_event.is_set():
        robot.send_action(base)
        precise_sleep(dt)


def hold_base_pose_for_seconds(robot, base_pose: Dict[str, float], hz: int, seconds: float) -> None:
    feats = set(robot.action_features.keys())
    base = fill_missing_with_current(robot, feats, normalize_action_keys(base_pose))
    dt = 1.0 / max(1, int(hz))
    end = time.perf_counter() + float(seconds)
    while time.perf_counter() < end:
        robot.send_action(base)
        precise_sleep(dt)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="Panbot/config/runtime.yaml", help="path to runtime.yaml")
    ap.add_argument("--repo_root", type=str, default="", help="optional: repo root path (auto if empty)")
    args = ap.parse_args()

    # repo root
    if args.repo_root.strip():
        repo_root = Path(args.repo_root).expanduser().resolve()
    else:
        # .../Panbot/control/main_runtime.py -> parents[2] == repo root (assuming Panbot is at repo_root/Panbot)
        repo_root = Path(__file__).resolve().parents[2]

    cfg_path = resolve_path(args.config, repo_root)
    cfg = load_yaml(cfg_path)

    log_path = setup_logging(repo_root / "Panbot" / "logs")
    logging.info("==================================================")
    logging.info("[main_runtime] START ✅")
    logging.info("repo_root=%s", repo_root)
    logging.info("config=%s", cfg_path)
    logging.info("logfile=%s", log_path)
    logging.info("==================================================")

    # resolve files
    corners_path = resolve_path(cfg["paths"]["corners"], repo_root)
    yolo_model_path = resolve_path(cfg["paths"]["yolo_model"], repo_root)
    gru_ckpt_path = resolve_path(cfg["paths"]["gru_ckpt"], repo_root)

    for p in [corners_path, yolo_model_path, gru_ckpt_path]:
        if not p.exists():
            raise FileNotFoundError(f"missing file: {p}")

    corners = load_corners(corners_path)
    warp_out = compute_warp_size(corners)
    logging.info("[WARP] corners=%s warp_out=%s", corners_path, warp_out)

    # unpack configs
    vision = cfg["vision"]
    yolo_cfg = cfg["yolo_trigger"]
    gru_cfg = cfg["gru_trigger"]
    task_cfg = cfg["task"]
    poses = cfg["poses"]
    policies = cfg["policies"]
    robot_cfg = cfg["robot"]

    base_pose = poses["base_pose"]
    task1_init = poses["task1_initial_sequence"]
    task1_ret = poses["task1_return_sequence"]

    # Build robot with policy cameras (B정석: robot 1번 connect, policy도 동일 robot 사용)
    so_cfg = build_so101_config(robot_cfg)
    robot = make_robot_from_config(so_cfg)

    # global fail-safe
    try:
        logging.info("[ROBOT] connecting port=%s id=%s", so_cfg.port, so_cfg.id)
        robot.connect()
        logging.info("[ROBOT] connected ✅ action_features=%d obs_features=%d",
                     len(robot.action_features), len(robot.observation_features))

        # --------------------
        # STAGE1: YOLO + Task1 (interrupt)
        # --------------------
        yolo_thread = YoloTriggerThread(
            cam_index=int(vision["cam_index"]),
            backend=str(vision["backend"]),
            width=int(vision["width"]),
            height=int(vision["height"]),
            fps=int(vision["fps"]),
            mjpg=bool(vision["mjpg"]),
            model_path=yolo_model_path,
            corners=corners,
            warp_out=warp_out,
            imgsz=int(yolo_cfg.get("imgsz", 640)),
            conf=float(yolo_cfg.get("conf", 0.25)),
            area_thr_ratio=float(yolo_cfg["area_thr_ratio"]),
            hold_frames=int(yolo_cfg["hold_frames"]),
            show=bool(vision["show"]),
            preview_scale=float(vision["yolo_preview_scale"]),
        )

        logging.info("[STAGE1] start YOLO thread + Task1")
        yolo_thread.start()

        run_task1_with_interrupt(
            robot=robot,
            hz=int(task_cfg["hz"]),
            base_pose=base_pose,
            initial_sequence=task1_init,
            return_sequence=task1_ret,
            ramp_time_s=float(task_cfg["task1_ramp_time_s"]),
            pose_hold_s=float(task_cfg["task1_pose_hold_s"]),
            interrupt_event=yolo_thread.trigger_event,
        )

        logging.info("[STAGE1] Task1 done. stopping YOLO thread...")
        yolo_thread.stop()

        if not yolo_thread.watchdog_ok():
            raise RuntimeError("YOLO watchdog failed (camera stalled)")

        # --------------------
        # STAGE2: GRU trigger while holding base
        # --------------------
        gru_thread = GruTriggerThread(
            cam_index=int(vision["cam_index"]),
            backend=str(vision["backend"]),
            width=int(vision["width"]),
            height=int(vision["height"]),
            fps=int(vision["fps"]),
            mjpg=bool(vision["mjpg"]),
            ckpt_path=gru_ckpt_path,
            corners=corners,
            warp_out=warp_out,
            preview_scale=float(vision["gru_preview_scale"]),
            show=bool(vision["show"]),
            image_size=int(gru_cfg["image_size"]),
            seq_len=int(gru_cfg["seq_len"]),
            stride=int(gru_cfg["stride"]),
            ema=float(gru_cfg["ema"]),
            ready_hold=int(gru_cfg["ready_hold"]),
            amp=bool(gru_cfg["amp"]),
        )

        logging.info("[STAGE2] start GRU thread + hold base")
        gru_thread.start()

        hold_base_pose_until(
            robot=robot,
            base_pose=base_pose,
            hz=int(task_cfg["hz"]),
            until_event=gru_thread.trigger_event,
        )

        logging.info("[STAGE2] GRU triggered. stopping GRU thread...")
        gru_thread.stop()

        if not gru_thread.watchdog_ok():
            raise RuntimeError("GRU watchdog failed (camera stalled)")

        # --------------------
        # STAGE3: Policy1 (Task2)
        # --------------------
        p1 = policies["policy1"]
        logging.info("[STAGE3] run policy1 repo=%s", p1["repo_id"])
        run_pretrained_policy_shared_robot(
            robot=robot,
            repo_id=str(p1["repo_id"]),
            fps=int(task_cfg["policy_fps"]),
            duration_s=float(task_cfg["task2_duration_s"]),
            task=p1.get("task", None),
            rename_map=p1.get("rename_map", {}) or {},
            dataset_repo_id=p1.get("dataset_repo_id", None),
            dataset_root=p1.get("dataset_root", None),
            use_amp=True,
        )

        logging.info("[STAGE3] policy1 done -> back to BASE")
        go_base_pose_best_effort(robot, base_pose, hz=int(task_cfg["hz"]), hold_s=1.0)

        # --------------------
        # WAIT: base for N sec
        # --------------------
        wait_s = float(task_cfg["wait_task2_to_task3_s"])
        logging.info("[WAIT] hold BASE for %.1fs", wait_s)
        hold_base_pose_for_seconds(robot, base_pose, hz=int(task_cfg["hz"]), seconds=wait_s)

        # --------------------
        # STAGE4: Policy2 (Task3)
        # --------------------
        p2 = policies["policy2"]
        logging.info("[STAGE4] run policy2 repo=%s", p2["repo_id"])
        run_pretrained_policy_shared_robot(
            robot=robot,
            repo_id=str(p2["repo_id"]),
            fps=int(task_cfg["policy_fps"]),
            duration_s=float(task_cfg["task3_duration_s"]),
            task=p2.get("task", None),
            rename_map=p2.get("rename_map", {}) or {},
            dataset_repo_id=p2.get("dataset_repo_id", None),
            dataset_root=p2.get("dataset_root", None),
            use_amp=True,
        )

        logging.info("[STAGE4] policy2 done -> back to BASE")
        go_base_pose_best_effort(robot, base_pose, hz=int(task_cfg["hz"]), hold_s=1.0)

        logging.info("[DONE] main_runtime finished OK ✅")

    except Exception as e:
        logging.exception("[FATAL] runtime exception: %s", e)
        # fail-safe: 어떤 에러든 base pose로 최대한 보내기
        try:
            go_base_pose_best_effort(robot, cfg["poses"]["base_pose"], hz=int(cfg["task"]["hz"]), hold_s=1.0)
        except Exception:
            pass
        raise

    finally:
        try:
            logging.info("[ROBOT] disconnect")
            robot.disconnect()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
