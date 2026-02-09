from __future__ import annotations

import argparse
import logging
import time
from enum import Enum, auto
from pathlib import Path

import cv2

from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

from Panbot.vision.modules.camera import open_camera, resize_for_preview
from Panbot.vision.modules.yoloseg_infer import YOLOSegInfer, YOLOSegConfig
from Panbot.vision.modules.gru_infer import GRUInfer, GRUInferConfig

from Panbot.tasks.base_pose import BasePoseController, HoldConfig
from Panbot.tasks.task1_motion import (
    Task1MotionStepper,
    Task1MotionConfig,
    DEFAULT_REST_ACTION,
)
from Panbot.tasks.policy1 import Policy1Runner, Policy1Config
from Panbot.tasks.policy2 import Policy2Runner, Policy2Config


class State(Enum):
    BASE_POSE_START = auto()

    TASK1_PREP_START_YOLO = auto()
    TASK1_RUN_INITIAL_AND_YOLO = auto()
    TASK1_HOLD_WAIT_YOLO = auto()
    TASK1_RUN_RETURN = auto()

    BASE_POSE_WAIT_GRU = auto()
    TASK2_POLICY = auto()
    WAIT_30S_BASE = auto()
    TASK3_POLICY = auto()
    FINAL_BASE_POSE = auto()
    DONE = auto()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def build_so101_robot(port: str, calib_dir: Path | None, robot_id: str):
    cfg = SO101FollowerConfig()
    cfg.port = port
    cfg.id = robot_id
    if calib_dir is not None:
        cfg.calibration_dir = calib_dir
    return cfg


def main():
    ap = argparse.ArgumentParser()

    # --- camera (4K 고정 기본값) ---
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--backend", type=str, default="v4l2")
    ap.add_argument("--mjpg", action="store_true", default=True)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--show", action="store_true", help="imshow 창 보기")
    ap.add_argument("--yolo_preview_scale", type=float, default=0.55)
    ap.add_argument("--gru_preview_scale", type=float, default=0.30)
    ap.add_argument("--base_preview_scale", type=float, default=0.30)

    # --- paths ---
    ap.add_argument("--corners", type=str, default="Panbot/vision/calibration/corners.json")
    ap.add_argument("--yolo_model", type=str, required=True)
    ap.add_argument("--gru_ckpt", type=str, required=True)

    # --- yolo thresholds ---
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--yolo_imgsz", type=int, default=640)
    ap.add_argument("--area_thr_ratio", type=float, default=0.17)
    ap.add_argument("--hold_frames", type=int, default=30)

    # --- gru params ---
    ap.add_argument("--gru_ready_hold", type=int, default=3)
    ap.add_argument("--gru_ema", type=float, default=0.7)
    ap.add_argument("--gru_amp", action="store_true", default=True)

    # --- robot ---
    ap.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    ap.add_argument("--robot_id", type=str, default="so101_follower_1")
    ap.add_argument("--robot_calib_dir", type=str, default="")

    # --- task2/task3 (임시: duration 기반) ---
    ap.add_argument("--task2_duration", type=float, default=10.0)
    ap.add_argument("--task3_duration", type=float, default=10.0)
    ap.add_argument("--wait_task2_to_task3_s", type=float, default=30.0)

    # --- base pose hold ---
    ap.add_argument("--base_hold_interval", type=float, default=0.25)
    ap.add_argument("--base_fps", type=int, default=30)

    # --- Task1 stepper fps ---
    ap.add_argument("--task1_fps", type=int, default=30)
    ap.add_argument("--task1_ramp_time_s", type=float, default=3.0)
    ap.add_argument("--task1_pose_hold_s", type=float, default=1.0)

    args = ap.parse_args()

    setup_logging()
    logging.info("=== Panbot main_runtime start ===")

    corners_path = Path(args.corners).expanduser().resolve()
    yolo_model_path = Path(args.yolo_model).expanduser().resolve()
    gru_ckpt_path = Path(args.gru_ckpt).expanduser().resolve()

    if not corners_path.exists():
        raise FileNotFoundError(f"corners not found: {corners_path}")
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"yolo model not found: {yolo_model_path}")
    if not gru_ckpt_path.exists():
        raise FileNotFoundError(f"gru checkpoint not found: {gru_ckpt_path}")

    robot_calib_dir = None
    if str(args.robot_calib_dir).strip():
        robot_calib_dir = Path(args.robot_calib_dir).expanduser().resolve()

    # --- open camera ---
    cap = open_camera(args.cam, args.backend, args.mjpg, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    # --- build & connect robot ---
    robot_cfg = build_so101_robot(args.robot_port, robot_calib_dir, args.robot_id)
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    logging.info("[ROBOT] connected port=%s id=%s", args.robot_port, args.robot_id)

    action_features = set(robot.action_features.keys())

    # --- controllers / configs ---
    base_pose = BasePoseController(
        robot=robot,
        cfg=HoldConfig(fps=args.base_fps, hold_interval_s=args.base_hold_interval, use_current_for_missing=True),
        action_features=action_features,
    )

    task1 = Task1MotionStepper(
        robot=robot,
        cfg=Task1MotionConfig(
            fps=args.task1_fps,
            ramp_time_s=args.task1_ramp_time_s,
            pose_hold_s=args.task1_pose_hold_s,
        ),
        action_features=action_features,
    )

    policy1 = Policy1Runner(robot=robot, cfg=Policy1Config(duration_s=args.task2_duration), action_features=action_features)
    policy2 = Policy2Runner(robot=robot, cfg=Policy2Config(duration_s=args.task3_duration), action_features=action_features)

    # vision modules (필요할 때 생성/삭제)
    yolo: YOLOSegInfer | None = None
    gru: GRUInfer | None = None

    # state vars
    state = State.BASE_POSE_START
    state_enter_t = time.perf_counter()

    # status log throttling
    last_status_log_t = 0.0
    STATUS_LOG_EVERY_S = 2.0

    def transition(next_state: State):
        nonlocal state, state_enter_t
        logging.info("[TRANSITION] %s -> %s", state.name, next_state.name)
        state = next_state
        state_enter_t = time.perf_counter()

    try:
        # 0) 시작: 기본자세
        base_pose.set_target(DEFAULT_REST_ACTION)
        base_pose.enable()

        while state != State.DONE:
            loop_start = time.perf_counter()
            now = loop_start

            # 30Hz 기준 frame read
            ok, frame = cap.read()
            if not ok or frame is None:
                logging.warning("[CAM] frame read fail")
                continue

            # visualization defaults
            vis = frame
            preview_scale = args.base_preview_scale

            # periodic status log
            if (now - last_status_log_t) >= STATUS_LOG_EVERY_S:
                last_status_log_t = now
                logging.info("[STATE] %s", state.name)

            # -----------------------
            # STATE MACHINE
            # -----------------------

            if state == State.BASE_POSE_START:
                base_pose.tick()
                if (now - state_enter_t) >= 1.0:
                    transition(State.TASK1_PREP_START_YOLO)

            elif state == State.TASK1_PREP_START_YOLO:
                # ✅ 방법 A + 인터럽트 대응:
                # 1) YOLO 먼저 "시작"(모델 로드/초기화)
                yolo = YOLOSegInfer(
                    YOLOSegConfig(
                        model_path=yolo_model_path,
                        conf=args.yolo_conf,
                        imgsz=args.yolo_imgsz,
                        use_warp=True,
                        corners_path=corners_path,
                        area_thr_ratio=args.area_thr_ratio,
                        hold_frames=args.hold_frames,
                    )
                )
                logging.info("[YOLO] started (before Task1 initial)")

                # 2) Task1 initial도 시작 (stepper)
                base_pose.disable()
                task1.start_initial()

                transition(State.TASK1_RUN_INITIAL_AND_YOLO)

            elif state == State.TASK1_RUN_INITIAL_AND_YOLO:
                # ✅ 여기서 "진짜 동시"로 돌아갑니다:
                # - task1.step()이 30Hz로 initial을 조금씩 진행
                # - yolo.step(frame)이 매 프레임 trigger 감시
                assert yolo is not None

                # Task1 1 tick 진행
                task1.step(now)

                # YOLO 1 tick 진행
                triggered, yvis, info = yolo.step(frame)
                vis = yvis
                preview_scale = args.yolo_preview_scale

                if triggered:
                    logging.info("[EVENT] YOLO trigger DURING initial/hold. info=%s", info)

                    # yolo는 이제 꺼도 됨(요구사항)
                    yolo = None
                    logging.info("[YOLO] stopped after trigger")

                    # ✅ initial 도중이면 즉시 return으로 전환
                    task1.interrupt_to_return()
                    transition(State.TASK1_RUN_RETURN)
                else:
                    # initial이 끝났는데 아직 trigger가 없으면 -> hold 단계로
                    if task1.is_initial_done():
                        last_pose = task1.get_last_pose_action()
                        if last_pose is None:
                            raise RuntimeError("Task1 initial done but last pose is None")

                        # hold는 base_pose로 유지(가볍게)
                        base_pose.set_target(last_pose)
                        base_pose.enable()

                        transition(State.TASK1_HOLD_WAIT_YOLO)

            elif state == State.TASK1_HOLD_WAIT_YOLO:
                # initial 마지막 포즈로 hold하며 yolo trigger 대기
                assert yolo is not None

                base_pose.tick()
                triggered, yvis, info = yolo.step(frame)
                vis = yvis
                preview_scale = args.yolo_preview_scale

                if triggered:
                    logging.info("[EVENT] YOLO trigger in HOLD. info=%s", info)
                    yolo = None
                    logging.info("[YOLO] stopped after trigger")

                    base_pose.disable()
                    task1.start_return()
                    transition(State.TASK1_RUN_RETURN)

            elif state == State.TASK1_RUN_RETURN:
                # return도 stepper로 진행 (blocking 아님)
                task1.step(now)

                # return 끝나면 기본자세로 복귀 + GRU 시작
                if task1.is_return_done():
                    base_pose.set_target(DEFAULT_REST_ACTION)
                    base_pose.enable()

                    gru = GRUInfer(
                        GRUInferConfig(
                            checkpoint_path=gru_ckpt_path,
                            use_warp=True,
                            corners_path=corners_path,
                            ema=args.gru_ema,
                            ready_hold=args.gru_ready_hold,
                            amp=args.gru_amp,
                        )
                    )
                    logging.info("[GRU] started (after Task1 return)")
                    transition(State.BASE_POSE_WAIT_GRU)

            elif state == State.BASE_POSE_WAIT_GRU:
                assert gru is not None
                base_pose.tick()

                triggered, gvis, info = gru.step(frame)
                vis = gvis
                preview_scale = args.gru_preview_scale

                if triggered:
                    logging.info("[EVENT] GRU trigger! info=%s", info)
                    gru = None
                    logging.info("[GRU] stopped after trigger")

                    base_pose.disable()
                    policy1.start()
                    transition(State.TASK2_POLICY)

            elif state == State.TASK2_POLICY:
                policy1.step(frame_bgr=frame)
                if policy1.is_done():
                    base_pose.set_target(DEFAULT_REST_ACTION)
                    base_pose.enable()
                    transition(State.WAIT_30S_BASE)
                policy1.sleep_to_fps(loop_start)

            elif state == State.WAIT_30S_BASE:
                base_pose.tick()
                remain = max(0.0, float(args.wait_task2_to_task3_s) - (now - state_enter_t))
                cv2.putText(vis, f"[WAIT] to TASK3: {remain:.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                if (now - state_enter_t) >= float(args.wait_task2_to_task3_s):
                    base_pose.disable()
                    policy2.start()
                    transition(State.TASK3_POLICY)

            elif state == State.TASK3_POLICY:
                policy2.step(frame_bgr=frame)
                if policy2.is_done():
                    base_pose.set_target(DEFAULT_REST_ACTION)
                    base_pose.enable()
                    transition(State.FINAL_BASE_POSE)
                policy2.sleep_to_fps(loop_start)

            elif state == State.FINAL_BASE_POSE:
                base_pose.tick()
                if (now - state_enter_t) >= 2.0:
                    logging.info("[DONE] back to base pose. exit.")
                    transition(State.DONE)

            # -----------------------
            # UI / quit
            # -----------------------
            if args.show:
                disp = resize_for_preview(vis, preview_scale)
                cv2.imshow("Panbot main_runtime", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logging.info("[USER] quit requested")
                    break

            # 30Hz 유지 (base_pose 쪽 sleep 사용)
            base_pose.sleep_to_fps(loop_start)

    finally:
        try:
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
        logging.info("=== Panbot main_runtime end ===")


if __name__ == "__main__":
    main()
