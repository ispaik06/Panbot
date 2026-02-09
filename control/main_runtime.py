# Panbot/control/main_runtime.py
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from vision.modules.camera import open_camera, resize_for_preview
from vision.modules.yolo_batter import YOLOBatterConfig, YOLOBatterInfer
from vision.modules.gru_bubble import GRUBubbleConfig, GRUBubbleInfer

from tasks.task1_motion import Task1MotionConfig, Task1Controller
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig


def run_policy_action():
    # TODO: 여기에 실제 policy 실행을 연결하세요.
    # 예: subprocess로 lerobot-eval 실행 or 파이썬 함수 호출
    logging.info("[POLICY] TODO: run policy here")


def main():
    ap = argparse.ArgumentParser()

    # project root = Panbot/
    ap.add_argument("--root", type=str, default="Panbot", help="project root folder name or path")

    # camera (4K fixed)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--backend", type=str, default="v4l2")
    ap.add_argument("--mjpg", action="store_true", default=True)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--fps", type=int, default=30)

    # corners (4K)
    ap.add_argument("--corners", type=str, default="vision/calibration/corners.json")

    # YOLO
    ap.add_argument("--yolo_model", type=str, default="vision/models/runs/batter_seg_local_v1/weights/best.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--yolo_imgsz", type=int, default=640)
    ap.add_argument("--yolo_area_thr_ratio", type=float, default=0.17)
    ap.add_argument("--yolo_hold_frames", type=int, default=30)

    # GRU
    ap.add_argument("--gru_ckpt", type=str, default="vision/models/runs/resnet18_gru16_cls/best.pt")
    ap.add_argument("--gru_ready_hold", type=int, default=3)
    ap.add_argument("--gru_ema", type=float, default=0.7)
    ap.add_argument("--gru_amp", action="store_true", default=True)

    # preview (state별로 다르게)
    ap.add_argument("--preview_scale_yolo", type=float, default=0.55)
    ap.add_argument("--preview_scale_gru", type=float, default=0.30)

    # show
    ap.add_argument("--show", action="store_true", default=True)

    # robot args (✅ main_runtime에서 받음)
    ap.add_argument("--robot_id", type=str, default="my_awesome_follower_arm")
    ap.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    ap.add_argument(
        "--robot_calibration_dir",
        type=str,
        default="/home/user/.cache/huggingface/lerobot/calibration/robots/so101_follower",
    )

    # task1 hold interval override (optional)
    ap.add_argument("--task1_hold_interval_s", type=float, default=0.25)
    ap.add_argument("--task1_ramp_time_s", type=float, default=3.0)

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    root = Path(args.root).resolve() if Path(args.root).exists() else Path.cwd() / args.root
    corners_path = (root / args.corners).resolve()

    # -------------------------
    # Vision modules
    # -------------------------
    yolo = YOLOBatterInfer(
        YOLOBatterConfig(
            model_path=(root / args.yolo_model).resolve(),
            conf=args.yolo_conf,
            imgsz=args.yolo_imgsz,
            use_warp=True,
            corners_path=corners_path,
            area_thr_ratio=args.yolo_area_thr_ratio,
            hold_frames=args.yolo_hold_frames,
        )
    )

    gru = GRUBubbleInfer(
        GRUBubbleConfig(
            checkpoint_path=(root / args.gru_ckpt).resolve(),
            corners_path=corners_path,
            use_warp=True,
            ema=args.gru_ema,
            ready_hold=args.gru_ready_hold,
            amp=bool(args.gru_amp),
        )
    )

    # -------------------------
    # Robot (Task1 controller)
    # -------------------------
    robot_cfg = SO101FollowerConfig(
        id=args.robot_id,
        port=args.robot_port,
        calibration_dir=Path(args.robot_calibration_dir),
    )

    task1_cfg = Task1MotionConfig(
        robot=robot_cfg,
        hold_interval_s=float(args.task1_hold_interval_s),
        ramp_time_s=float(args.task1_ramp_time_s),
    )
    task1 = Task1Controller(task1_cfg)

    # -------------------------
    # Camera open ONCE (4K fixed)
    # -------------------------
    cap = open_camera(args.cam, args.backend, True, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    # state machine
    state = "WAIT_YOLO"
    logging.info("[FLOW] START -> WAIT_YOLO")

    yolo.reset()
    gru.reset()

    last_state_print = time.time()

    try:
        # ✅ 로봇 연결 + Task1 시작 시퀀스 실행
        task1.connect()
        task1.start()

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # ✅ 어떤 상태든 Task1 hold 유지 (정석 B)
            task1.hold_tick()

            if state == "WAIT_YOLO":
                triggered, vis, info = yolo.step(frame)
                preview_scale = float(args.preview_scale_yolo)

                if time.time() - last_state_print > 2.0:
                    logging.info("[STATE] WAIT_YOLO ratio=%.3f hit=%d/%d trig=%s",
                                 info["ratio"], info["hit_count"], info["hold_frames"], info["triggered"])
                    last_state_print = time.time()

                if triggered:
                    logging.info("[FLOW] YOLO_TRIGGER -> Task1 RETURN")
                    task1.do_return()

                    gru.reset()
                    state = "WAIT_GRU"
                    logging.info("[FLOW] now -> WAIT_GRU")

            elif state == "WAIT_GRU":
                triggered, vis, info = gru.step(frame)
                preview_scale = float(args.preview_scale_gru)

                if time.time() - last_state_print > 2.0:
                    logging.info("[STATE] WAIT_GRU label=%s conf=%.2f hold=%d/%d trig=%s",
                                 info["label"], info["conf"], info["ready_streak"], info["ready_hold"], info["triggered"])
                    last_state_print = time.time()

                if triggered:
                    logging.info("[FLOW] GRU_READY -> RUN_POLICY")
                    run_policy_action()
                    state = "RUN_POLICY"
                    logging.info("[FLOW] now -> RUN_POLICY")

            else:  # RUN_POLICY
                vis = frame
                preview_scale = 0.35
                if time.time() - last_state_print > 2.0:
                    logging.info("[STATE] RUN_POLICY ... (TODO: implement)")
                    last_state_print = time.time()

            if args.show:
                disp = resize_for_preview(vis, preview_scale)
                cv2.imshow("Panbot Runtime (single camera 4K)", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

                # 디버그 전환
                if key == ord("1"):
                    logging.info("[DEBUG] force WAIT_YOLO")
                    yolo.reset()
                    state = "WAIT_YOLO"
                if key == ord("2"):
                    logging.info("[DEBUG] force WAIT_GRU")
                    gru.reset()
                    state = "WAIT_GRU"

    finally:
        cap.release()
        cv2.destroyAllWindows()
        task1.disconnect()
        logging.info("[DONE]")


if __name__ == "__main__":
    main()
