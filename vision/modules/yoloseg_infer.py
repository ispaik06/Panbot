from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO


def load_corners(corners_path: Path) -> np.ndarray:
    data = json.loads(corners_path.read_text(encoding="utf-8"))
    pts = data.get("points", None)
    if not pts or len(pts) != 4:
        raise ValueError("corners.json must contain 'points' with 4 entries (TL,TR,BR,BL)")
    return np.array([[float(p["x"]), float(p["y"])] for p in pts], dtype=np.float32)


def compute_warp_size_from_corners(corners: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = corners
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    out_w = int(round(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    out_h = int(round(max(height_a, height_b)))

    out_w = max(out_w, 50)
    out_h = max(out_h, 50)
    return out_w, out_h


def warp_topview(frame_bgr: np.ndarray, corners: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame_bgr, M, (out_w, out_h))


def pick_largest_mask(masks_bool: np.ndarray) -> np.ndarray:
    areas = masks_bool.reshape(masks_bool.shape[0], -1).sum(axis=1)
    idx = int(np.argmax(areas))
    return masks_bool[idx]


def overlay_mask(frame_bgr: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if mask_bool.shape[0] != h or mask_bool.shape[1] != w:
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_u8 > 0

    overlay = frame_bgr.copy()
    overlay[mask_bool] = (0, 200, 0)
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)


@dataclass
class YOLOSegConfig:
    model_path: Path
    conf: float = 0.25
    imgsz: int = 640

    use_warp: bool = True
    corners_path: Optional[Path] = None
    warp_w: int = 0
    warp_h: int = 0

    area_thr_ratio: float = 0.17
    hold_frames: int = 30

    overlay_alpha: float = 0.45


class YOLOSegInfer:
    def __init__(self, cfg: YOLOSegConfig):
        self.cfg = cfg
        self.model = YOLO(str(cfg.model_path))

        self.corners = None
        self.warp_out_w = None
        self.warp_out_h = None

        if cfg.use_warp:
            if cfg.corners_path is None:
                raise ValueError("use_warp=True but corners_path is None")
            self.corners = load_corners(cfg.corners_path)

            if cfg.warp_w > 0 and cfg.warp_h > 0:
                self.warp_out_w = int(cfg.warp_w)
                self.warp_out_h = int(cfg.warp_h)
            else:
                w, h = compute_warp_size_from_corners(self.corners)
                self.warp_out_w, self.warp_out_h = int(w), int(h)

        self.reset()

    def reset(self):
        self.hit_count = 0
        self.stable_trigger = False

    def step(self, frame_bgr: np.ndarray) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        frame = frame_bgr
        if self.cfg.use_warp:
            frame = warp_topview(frame, self.corners, self.warp_out_w, self.warp_out_h)

        H, W = frame.shape[:2]
        frame_area = float(H * W)

        results = self.model.predict(frame, imgsz=self.cfg.imgsz, conf=self.cfg.conf, verbose=False)
        r = results[0]

        mask_area = 0.0
        ratio = 0.0
        triggered = False
        vis = frame

        if r.masks is not None and r.masks.data is not None and len(r.masks.data) > 0:
            masks = r.masks.data.detach().cpu().numpy()  # (N,h,w) float 0..1
            masks_bool = masks > 0.5
            largest = pick_largest_mask(masks_bool)

            if largest.shape[0] != H or largest.shape[1] != W:
                largest_u8 = (largest.astype(np.uint8) * 255)
                largest_u8 = cv2.resize(largest_u8, (W, H), interpolation=cv2.INTER_NEAREST)
                largest = largest_u8 > 0

            mask_area = float(largest.sum())
            ratio = mask_area / frame_area

            if ratio >= float(self.cfg.area_thr_ratio):
                self.hit_count += 1
            else:
                self.hit_count = 0

            self.stable_trigger = self.hit_count >= int(self.cfg.hold_frames)
            triggered = bool(self.stable_trigger)

            vis = overlay_mask(frame, largest, alpha=self.cfg.overlay_alpha)

        text = f"[YOLO] ratio={ratio:.3f} thr={self.cfg.area_thr_ratio:.3f} hit={self.hit_count}/{self.cfg.hold_frames} TRIG={triggered}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        info = {
            "ratio": float(ratio),
            "mask_area": float(mask_area),
            "hit_count": int(self.hit_count),
            "hold_frames": int(self.cfg.hold_frames),
            "triggered": bool(triggered),
        }
        return triggered, vis, info
