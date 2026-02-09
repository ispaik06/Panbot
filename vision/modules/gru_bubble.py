# Panbot/vision/modules/gru_bubble.py
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18


DEFAULT_LABELS = ["not_ready", "almost_ready", "ready"]


class ResNet18GRU(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 1, num_classes: int = 3):
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
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.backbone(x).flatten(1)   # (B*T,512)
        feat = feat.reshape(B, T, -1)        # (B,T,512)
        out, _ = self.gru(feat)
        last = out[:, -1, :]
        return self.head(last)


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
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    return max(maxW, 2), max(maxH, 2)


def warp_frame(frame_bgr: np.ndarray, corners: np.ndarray, out_w: Optional[int], out_h: Optional[int]) -> np.ndarray:
    if out_w is None or out_h is None:
        w, h = compute_warp_size(corners)
        if out_w is None:
            out_w = w
        if out_h is None:
            out_h = h
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame_bgr, M, (int(out_w), int(out_h)), flags=cv2.INTER_LINEAR)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess_frame_bgr(frame_bgr: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device, non_blocking=True)


def draw_hud(vis_bgr: np.ndarray, line1: str, line2: str):
    h, w = vis_bgr.shape[:2]
    ref = 720.0
    s = min(w, h) / ref
    s = float(np.clip(s, 0.60, 2.80))

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs1 = 1.0 * s
    fs2 = 0.80 * s
    th1 = max(2, min(int(round(2 * s)), 8))
    th2 = max(2, min(int(round(2 * s)), 8))

    mx = int(round(12 * s))
    my = int(round(18 * s))

    (tw1, thh1), _ = cv2.getTextSize(line1, font, fs1, th1)
    (tw2, thh2), _ = cv2.getTextSize(line2, font, fs2, th2)
    gap = max(6, int(round(12 * s))) + int(round(0.25 * max(thh1, thh2)))

    y1 = my + thh1
    y2 = y1 + gap + thh2

    def put_text(text, x, y, fs, thick):
        cv2.putText(vis_bgr, text, (x + 2, y + 2), font, fs, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(vis_bgr, text, (x, y), font, fs, (0, 255, 255), thick, cv2.LINE_AA)

    put_text(line1, mx, y1, fs1, th1)
    put_text(line2, mx, y2, fs2, th2)


@dataclass
class GRUBubbleConfig:
    checkpoint_path: Path
    corners_path: Optional[Path] = None
    use_warp: bool = True
    warp_w: int = 0
    warp_h: int = 0

    image_size: int = 224
    seq_len: int = 16
    stride: int = 6

    ema: float = 0.7
    ready_hold: int = 3

    amp: bool = True


class GRUBubbleInfer:
    """
    step(frame_bgr) -> (triggered, vis, info)
    - triggered: ready confirmed (ready_hold consecutive after EMA)
    """

    def __init__(self, cfg: GRUBubbleConfig):
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[GRU] device:", self.device)

        ckpt = torch.load(str(cfg.checkpoint_path), map_location="cpu")
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise ValueError("Checkpoint format invalid (expected dict with key 'model').")

        label2id = ckpt.get("label2id", {name: i for i, name in enumerate(DEFAULT_LABELS)})
        self.id2label = {int(v): k for k, v in label2id.items()}
        self.label2id = {k: int(v) for k, v in label2id.items()}
        num_classes = len(self.label2id)

        ckpt_args = ckpt.get("args", {})
        hidden = int(ckpt_args.get("hidden", 256))
        gru_layers = int(ckpt_args.get("gru_layers", 1))

        self.model = ResNet18GRU(hidden_size=hidden, num_layers=gru_layers, num_classes=num_classes)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        print("[GRU] ckpt:", cfg.checkpoint_path)
        print("[GRU] classes:", num_classes, "labels:", [self.id2label[i] for i in range(num_classes)])

        # warp
        self.corners = None
        self.warp_w = None
        self.warp_h = None
        if cfg.use_warp:
            if cfg.corners_path is None:
                raise ValueError("use_warp=True but corners_path is None")
            self.corners = load_corners(cfg.corners_path)
            self.warp_w = cfg.warp_w if cfg.warp_w > 0 else None
            self.warp_h = cfg.warp_h if cfg.warp_h > 0 else None
            print("[GRU] warp ON corners=", cfg.corners_path, "size=", (self.warp_w or "auto"), "x", (self.warp_h or "auto"))
        else:
            print("[GRU] warp OFF")

        # buffer
        self.need = (cfg.seq_len - 1) * cfg.stride + 1
        self.buf = deque(maxlen=self.need)

        self.ema_prob = None
        self.ready_id = self.label2id.get("ready", num_classes - 1)
        self.ready_streak = 0
        self.infer_count = 0
        self.t0 = time.time()

    def reset(self):
        self.buf.clear()
        self.ema_prob = None
        self.ready_streak = 0
        self.infer_count = 0
        self.t0 = time.time()

    def step(self, frame_bgr: np.ndarray):
        # warp (optional)
        if self.corners is not None:
            frame_proc = warp_frame(frame_bgr, self.corners, self.warp_w, self.warp_h)
        else:
            frame_proc = frame_bgr

        self.buf.append(frame_proc)

        pred_label = "warming_up"
        conf = 0.0
        triggered = False

        if len(self.buf) == self.need:
            indices = [self.need - 1 - self.cfg.stride * (self.cfg.seq_len - 1 - i) for i in range(self.cfg.seq_len)]
            frames = [self.buf[idx] for idx in indices]

            xs = [preprocess_frame_bgr(f, self.cfg.image_size, self.device) for f in frames]
            x = torch.cat(xs, dim=0).unsqueeze(0)  # (1,T,C,H,W)

            use_amp = bool(self.cfg.amp) and (self.device.type == "cuda")
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        logits = self.model(x)
                else:
                    logits = self.model(x)

            prob = torch.softmax(logits, dim=1).float().detach().cpu().numpy()[0]

            if self.ema_prob is None:
                self.ema_prob = prob
            else:
                self.ema_prob = float(self.cfg.ema) * self.ema_prob + (1.0 - float(self.cfg.ema)) * prob

            pred_id = int(np.argmax(self.ema_prob))
            conf = float(self.ema_prob[pred_id])

            if pred_id == self.ready_id:
                self.ready_streak += 1
            else:
                self.ready_streak = 0

            pred_label = self.id2label.get(pred_id, str(pred_id))
            triggered = (pred_id == self.ready_id) and (self.ready_streak >= int(self.cfg.ready_hold))

            self.infer_count += 1

        # HUD
        dt = time.time() - self.t0
        fps_est = (self.infer_count / dt) if dt > 0 else 0.0

        line1 = f"[GRU] {pred_label} conf={conf:.2f} hold={self.ready_streak}/{self.cfg.ready_hold} TRIGGER={triggered}"
        line2 = f"buf={len(self.buf)}/{self.need} infer={self.infer_count} fps~{fps_est:.1f}"

        vis = frame_proc.copy()
        draw_hud(vis, line1, line2)

        info: Dict[str, Any] = {
            "label": pred_label,
            "conf": conf,
            "ready_streak": self.ready_streak,
            "ready_hold": int(self.cfg.ready_hold),
            "triggered": bool(triggered),
        }
        return triggered, vis, info
