from __future__ import annotations

import cv2


def backend_to_cv2(backend: str) -> int:
    b = (backend or "auto").lower().strip()
    if b in ("", "auto", "any"):
        return 0
    if b in ("v4l2", "cap_v4l2"):
        return cv2.CAP_V4L2
    if b in ("gstreamer", "gst", "cap_gstreamer"):
        return cv2.CAP_GSTREAMER
    if b in ("ffmpeg", "cap_ffmpeg"):
        return cv2.CAP_FFMPEG
    if b in ("msmf", "cap_msmf"):
        return cv2.CAP_MSMF
    if b in ("dshow", "cap_dshow"):
        return cv2.CAP_DSHOW
    raise ValueError(f"Unknown backend: {backend}")


def open_camera(cam_index: int, backend: str, mjpg: bool, width: int, height: int, fps: int) -> cv2.VideoCapture:
    api = backend_to_cv2(backend)
    cap = cv2.VideoCapture(cam_index, api)

    if not cap.isOpened():
        return cap

    # Apply settings (best effort)
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))

    # Read back actual
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    print(f"[CAM] index={cam_index} backend={backend} mjpg={mjpg}")
    print(f"[CAM] requested {width}x{height}@{fps} -> actual {actual_w}x{actual_h}@{actual_fps:.2f} fourcc={fourcc}")
    return cap


def resize_for_preview(img, scale: float):
    if scale <= 0 or abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
