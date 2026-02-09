# Panbot/control/main_runtime.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from vision.modules.camera import open_camera, resize_for_preview
from vision.modules.yolo_batter import YOLOBatterConfig, YOLOBatterInfer
from vision.modules.gru_bubble import GRUBubbleConfig, GRUBubbleInfer


def task1_return_action():
    # TODO: 여기에 실제 Task1 return 동작을 연결하세요.
    # 예: task1.run_return()
    print("[TASK] Task1 return action (TODO)")


def run_policy_action():
    # TODO: 여기에 실제 policy 실행을 연결하세요.
    # 예: subprocess로 lerobot-eval 실행 or 파이썬 함수 호출
    print("[TASK] Run policy (TODO)")


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
    args = ap.parse_args()

    root = Path(args.root).resolve() if Path(args.root).exists() else Path.cwd() / args.root
    corners_path = (root / args.corners).resolve()

    # init infer modules
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

    # open camera ONCE (4K fixed)
    cap = open_camera(args.cam, args.backend, True, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    # state machine
    # Flow:
    # START -> WAIT_YOLO -> (YOLO trigger) -> Task1 return -> WAIT_GRU -> (GRU trigger) -> RUN_POLICY
    state = "WAIT_YOLO"
    print("[FLOW] START -> WAIT_YOLO")

    # reset internal states
    yolo.reset()
    gru.reset()

    last_state_print = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            if state == "WAIT_YOLO":
                triggered, vis, info = yolo.step(frame)
                preview_scale = float(args.preview_scale_yolo)

                if time.time() - last_state_print > 2.0:
                    print(f"[STATE] WAIT_YOLO ratio={info['ratio']:.3f} hit={info['hit_count']}/{info['hold_frames']} trig={info['triggered']}")
                    last_state_print = time.time()

                if triggered:
                    print("[FLOW] YOLO_TRIGGER -> Task1 return")
                    task1_return_action()

                    # 다음 단계로 넘어갈 때 GRU 상태 초기화 권장
                    gru.reset()
                    state = "WAIT_GRU"
                    print("[FLOW] now -> WAIT_GRU")

            elif state == "WAIT_GRU":
                triggered, vis, info = gru.step(frame)
                preview_scale = float(args.preview_scale_gru)

                if time.time() - last_state_print > 2.0:
                    print(f"[STATE] WAIT_GRU label={info['label']} conf={info['conf']:.2f} hold={info['ready_streak']}/{info['ready_hold']} trig={info['triggered']}")
                    last_state_print = time.time()

                if triggered:
                    print("[FLOW] GRU_READY -> RUN_POLICY")
                    run_policy_action()
                    state = "RUN_POLICY"
                    print("[FLOW] now -> RUN_POLICY")

            else:  # RUN_POLICY
                # policy를 “블로킹으로 실행”하면 여기 루프가 잠깐 멈출 수 있음.
                # 필요하면 여기서 policy 상태 관리/중단키 등을 확장하면 됩니다.
                vis = frame
                preview_scale = 0.35
                if time.time() - last_state_print > 2.0:
                    print("[STATE] RUN_POLICY ... (TODO: implement)")
                    last_state_print = time.time()

            if args.show:
                disp = resize_for_preview(vis, preview_scale)
                cv2.imshow("Panbot Runtime (A: single camera)", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                # 디버그용 강제 전환 키
                if key == ord("1"):
                    print("[DEBUG] force WAIT_YOLO")
                    yolo.reset()
                    state = "WAIT_YOLO"
                if key == ord("2"):
                    print("[DEBUG] force WAIT_GRU")
                    gru.reset()
                    state = "WAIT_GRU"

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[DONE]")


if __name__ == "__main__":
    main()
