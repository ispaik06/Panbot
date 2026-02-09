# vision/scripts/gru_infer.py
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torchvision.models import resnet18


# ==================================================
# Config loader
# ==================================================
def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config():
    root = resolve_project_root()
    cfg_path = root / "config" / "default.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg, root


# ==================================================
# Model
# ==================================================
class ResNet18GRU(nn.Module):
    def __init__(self, hidden=256, layers=1, classes=3):
        super().__init__()
        backbone = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.gru = nn.GRU(512, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.backbone(x).flatten(1)
        feat = feat.reshape(B, T, -1)
        out, _ = self.gru(feat)
        return self.head(out[:, -1])


# ==================================================
# Warp utils
# ==================================================
def load_corners(path: Path):
    data = json.loads(path.read_text())
    pts = data["points"]
    return np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)


def compute_warp_size(corners):
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    return max(w, 2), max(h, 2)


def warp(frame, corners, w, h):
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame, M, (w, h))


# ==================================================
# Preprocess
# ==================================================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess(frame, size, device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    x = torch.from_numpy(rgb).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


# ==================================================
# Main
# ==================================================
def main():
    cfg, root = load_config()

    cam_shared = cfg["camera"]
    task = cfg["gru"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    ckpt = torch.load(root / task["model"]["checkpoint"], map_location="cpu")
    label2id = ckpt.get("label2id", {"not_ready": 0, "almost_ready": 1, "ready": 2})
    id2label = {i: k for k, i in label2id.items()}

    hidden = ckpt.get("args", {}).get("hidden", 256)
    layers = ckpt.get("args", {}).get("gru_layers", 1)

    model = ResNet18GRU(hidden, layers, len(label2id))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # warp
    corners = load_corners(root / task["calibration"]["corners"])
    warp_w, warp_h = compute_warp_size(corners)

    # camera
    cap = cv2.VideoCapture(cam_shared["cam"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, task["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, task["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_shared["fps"])
    if cam_shared["mjpg"]:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    need = (task["sequence"]["seq_len"] - 1) * task["sequence"]["stride"] + 1
    buf = deque(maxlen=need)

    ema_prob = None
    ready_id = label2id["ready"]
    ready_streak = 0
    infer_count = 0
    t0 = time.time()

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = warp(frame, corners, warp_w, warp_h)
            buf.append(frame)

            label = "warming_up"
            conf = 0.0

            if len(buf) == need:
                idxs = [
                    need - 1 - task["sequence"]["stride"] * (task["sequence"]["seq_len"] - 1 - i)
                    for i in range(task["sequence"]["seq_len"])
                ]
                xs = [preprocess(buf[i], task["model"]["image_size"], device) for i in idxs]
                x = torch.cat(xs).unsqueeze(0)

                with torch.cuda.amp.autocast(enabled=(task["amp"] and device.type == "cuda")):
                    logits = model(x)
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

                ema = task["smoothing"]["ema"]
                ema_prob = prob if ema_prob is None else ema * ema_prob + (1 - ema) * prob

                pred_id = int(np.argmax(ema_prob))
                conf = float(ema_prob[pred_id])

                ready_streak = ready_streak + 1 if pred_id == ready_id else 0
                if pred_id == ready_id and ready_streak < task["smoothing"]["ready_hold"]:
                    label = "almost_ready"
                else:
                    label = id2label[pred_id]

                infer_count += 1

            scale = task["preview"]["scale"]
            disp = cv2.resize(
                frame,
                (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
            )

            fps = infer_count / max(time.time() - t0, 1e-6)
            cv2.putText(disp, f"{label} conf={conf:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(disp, f"fps~{fps:.1f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("gru_infer", disp)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
