# vision/scripts/yoloseg_infer.py
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


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
    return max(w, 50), max(h, 50)


def warp(frame, corners, w, h):
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame, M, (w, h))


# ==================================================
# Main
# ==================================================
def main():
    cfg, root = load_config()

    cam_shared = cfg["camera"]
    task = cfg["yoloseg"]

    # model
    model = YOLO(str(root / task["model"]["path"]))

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

    hit = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = warp(frame, corners, warp_w, warp_h)
        H, W = frame.shape[:2]

        r = model.predict(
            frame,
            imgsz=task["model"]["imgsz"],
            conf=task["model"]["conf"],
            verbose=False,
        )[0]

        ratio = 0.0
        if r.masks is not None:
            mask = (r.masks.data[0].cpu().numpy() > 0.5)
            ratio = mask.sum() / (H * W)
            hit = hit + 1 if ratio >= task["area"]["thr_ratio"] else 0

        triggered = hit >= task["area"]["hold_frames"]

        cv2.putText(
            frame,
            f"ratio={ratio:.3f}  TRIGGER={triggered}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

        scale = task["preview"]["scale"]
        disp = cv2.resize(frame, (int(W * scale), int(H * scale)))

        cv2.imshow("yoloseg_infer", disp)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
