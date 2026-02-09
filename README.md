
# Panbot 🤖🥞

Vision-triggered robot runtime for SO101 follower arm + YOLO segmentation + GRU readiness trigger + LeRobot policies.

## What this project does

`Panbot/control/main_runtime.py`는 아래 순서로 동작합니다.

1. **로봇 연결** (SO101 follower)
2. **Vision 카메라 오픈** (YOLO/GRU가 공유하는 단일 카메라)
3. **Task1 모션 실행** 중 YOLO 트리거가 발생하면 복귀(return) 시퀀스로 전환
4. Task1 return 완료 후 **GRU 트리거 대기**
5. GRU 트리거 발생 시 **Policy1 실행 (Task2)**
6. 대기 후 **Policy2 실행 (Task3)**
7. 마지막에 **Base pose 복귀 + 안전 종료**

> 핵심:
>
> * **vision.cam_index 카메라**는 YOLO/GRU 트리거용(단일 공유)
> * **robot.cameras**는 Policy(LeRobot) 관측용(오른/왼/글로벌/손목 등)로 별도입니다.

---

## Folder Structure

```bash
Panbot/
├─ config/
│  └─ runtime.yaml                  # 런타임 설정(카메라/트리거/태스크/정책/포즈)
│
├─ control/
│  └─ main_runtime.py               # ✅ 메인 실행 파일(전체 파이프라인 오케스트레이션)
│
├─ vision/
│  ├─ calibration/
│  │  └─ corners.json               # 워프(원근 보정)용 4점 코너
│  │
│  ├─ models/
│  │  └─ runs/
│  │     ├─ batter_seg_local_v1/weights/best.pt     # YOLO Seg 모델
│  │     └─ resnet18_gru16_cls/best.pt              # GRU 체크포인트
│  │
│  └─ modules/
│     ├─ camera.py                  # open_camera(), resize_for_preview()
│     ├─ yoloseg_infer.py            # YOLOSegConfig, YOLOSegInfer (trigger+vis)
│     └─ gru_infer.py                # GRUInferConfig, GRUInfer (trigger+vis)
│
├─ tasks/
│  ├─ base_pose.py                   # BasePoseController, HoldConfig
│  └─ task1_motion.py                # Task1MotionConfig/Stepper, DEFAULT_REST_ACTION
│
├─ policies/
│  └─ common_policy_runner.py        # run_pretrained_policy_shared_robot()
│
└─ logs/
   └─ (runtime logs output here)     # runtime.yaml의 log.dir 기준
```

> 실제 경로는 프로젝트 상태에 따라 조금 다를 수 있으나, `main_runtime.py`가 직접 import하는 경로는 위 구조를 기준으로 합니다.

---

## Main Entry: `Panbot/control/main_runtime.py`

### What it imports & uses (with paths)

#### 1) Runtime Config / Logging

* **Config**

  * `Panbot/config/runtime.yaml`
* **Logging**

  * runtime.yaml의 `log.dir`, `log.level`을 읽어서 파일+stdout 로깅

관련 코드:

* `_load_yaml()`, `_normalize_runtime_config()`
* `_setup_logging(log_dir, level)`

---

#### 2) Robot (SO101 follower)

* 로봇 구성/생성:

  * `from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig`
  * `from lerobot.robots import make_robot_from_config`
* 로봇 config 생성 함수:

  * `_build_so101_config(cfg["robot"])`

로봇 관련 runtime.yaml 키:

* `robot.port`, `robot.id`, `robot.calibration_dir`
* `robot.cameras.*` (Policy 관측용 카메라들)

---

#### 3) Vision Camera (YOLO/GRU shared)

* Vision 카메라 오픈:

  * `from Panbot.vision.modules.camera import open_camera, resize_for_preview`
* runtime.yaml에서 읽는 값:

  * `vision.cam_index`, `vision.backend`, `vision.mjpg`
  * `vision.width`, `vision.height`, `vision.fps`

실제 적용되는 부분:

```python
cap = open_camera(
  cam_index=cam_index,
  backend=backend,
  mjpg=mjpg,
  width=width,
  height=height,
  fps=fps,
)
```

---

#### 4) YOLO Trigger

* 파일:

  * `Panbot/vision/modules/yoloseg_infer.py`
* 클래스:

  * `YOLOSegConfig`, `YOLOSegInfer`

runtime.yaml 키:

* `yolo_trigger.conf`, `imgsz`
* `yolo_trigger.area_thr_ratio`, `hold_frames`
* `yolo_trigger.use_warp`, `warp_w`, `warp_h`
* 워프 코너 파일: `paths.corners`

동작:

* 루프에서 `yolo.step(frame)` 호출 → `(trig, vis, info)` 반환
* `trig=True`가 최초 발생하면 Task1을 return으로 전환

---

#### 5) GRU Trigger

* 파일:

  * `Panbot/vision/modules/gru_infer.py`
* 클래스:

  * `GRUInferConfig`, `GRUInfer`

runtime.yaml 키:

* `gru_trigger.image_size`, `seq_len`, `stride`
* `gru_trigger.ema`, `ready_hold`, `amp`
* `gru_trigger.use_warp`, `warp_w`, `warp_h`

동작:

* Task1 return 종료 후 `gru.reset()`
* 루프에서 `gru.step(frame)` 호출 → `(trig, vis, info)`
* `trig=True`면 Policy 단계로 넘어감

---

#### 6) Task1 Motion (Robot motion)

* 파일:

  * `Panbot/tasks/task1_motion.py`
* 클래스:

  * `Task1MotionConfig`, `Task1MotionStepper`
* base pose 관련:

  * `DEFAULT_REST_ACTION`

runtime.yaml 키:

* `task.task1_ramp_time_s`
* `task.task1_pose_hold_s`
* `poses.task1_initial_sequence`
* `poses.task1_return_sequence`

로봇이 실제로 움직이는 핵심 호출:

* `task1.start_initial()`
* 루프에서 `task1.step(time.perf_counter())`
* 트리거 시 `task1.interrupt_to_return()`

---

#### 7) Base Pose Controller (keep stable)

* 파일:

  * `Panbot/tasks/base_pose.py`
* 클래스:

  * `BasePoseController`, `HoldConfig`

runtime.yaml 키:

* `poses.base_pose`
* `task.base_pose_hold_interval_s`

사용 목적:

* Task1/Policy 사이 구간에서 로봇을 안정적으로 base pose로 유지
* `base_ctrl.tick()`이 호출되는 동안 유지됨

---

#### 8) Policies (LeRobot pretrained)

* 파일:

  * `Panbot/policies/common_policy_runner.py`
* 함수:

  * `run_pretrained_policy_shared_robot(...)`

runtime.yaml 키:

* `task.policy_fps`
* `task.task2_duration_s` (policy1 duration)
* `task.task3_duration_s` (policy2 duration)
* `task.wait_task2_to_task3_s`
* `policies.policy1.repo_id`
* `policies.policy2.repo_id`
* `policies.*.use_amp`, `print_joints`, `print_joints_every` 등

로봇이 실제로 움직이는 핵심(Policy 단계):

* `common_policy_runner.py` 내부의 `robot.send_action(...)`

---

## Configuration: `Panbot/config/runtime.yaml`

### Required paths

```yaml
paths:
  corners: "Panbot/vision/calibration/corners.json"
  yolo_model: "Panbot/vision/models/runs/.../best.pt"
  gru_ckpt: "Panbot/vision/models/runs/.../best.pt"
```

### Vision camera (YOLO/GRU shared)

```yaml
vision:
  cam_index: 0
  backend: "v4l2"
  mjpg: true
  width: 3840
  height: 2160
  fps: 30
  show: true
  yolo_preview_scale: 0.55
  gru_preview_scale: 0.30
  watchdog_s: 2.0
```

### Robot & policy observation cameras (separate from vision cam)

```yaml
robot:
  port: "/dev/ttyACM0"
  id: "my_awesome_follower_arm"
  cameras:
    right: { type: "opencv", index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG" }
    ...
```

### Triggers

```yaml
yolo_trigger:
  conf: 0.25
  imgsz: 640
  area_thr_ratio: 0.17
  hold_frames: 30
  use_warp: true
  warp_w: 0
  warp_h: 0
```

```yaml
gru_trigger:
  image_size: 224
  seq_len: 16
  stride: 6
  ema: 0.7
  ready_hold: 3
  amp: true
  use_warp: true
  warp_w: 0
  warp_h: 0
```

### Task / Timing / Policies

```yaml
task:
  hz: 30
  task1_ramp_time_s: 3.0
  task1_pose_hold_s: 1.0
  base_pose_hold_interval_s: 0.25
  policy_fps: 30
  task2_duration_s: 10.0
  task3_duration_s: 10.0
  wait_task2_to_task3_s: 30.0
```

---

## Quick Start

### 1) Install / Environment

* Python 환경 + LeRobot + OpenCV + Torch(CUDA) 등이 설치되어 있어야 합니다.
* CUDA가 있는 환경을 가정합니다.

### 2) Check camera indices

* `vision.cam_index`는 **YOLO/GRU 트리거용 단일 카메라**입니다.
* `robot.cameras.*.index_or_path`는 policy observation 카메라입니다.

### 3) Run

```bash
python Panbot/control/main_runtime.py --config Panbot/config/runtime.yaml
```

---

## Runtime Flow (debug-friendly)

* **Stage1:** Task1 initial + YOLO trigger

  * `task1.step()`이 로봇을 움직입니다.
* **Stage2:** Base pose 유지 + GRU trigger

  * `base_ctrl.tick()`이 로봇을 base pose에 붙잡아둡니다.
* **Stage3:** Policy1 실행

  * policy runner 내부 `robot.send_action()`이 로봇을 움직입니다.
* **Wait**
* **Stage4:** Policy2 실행

---

## Logs

* 저장 위치: `log.dir` (기본 `Panbot/logs`)
* 파일명 예: `main_runtime_YYYYMMDD_HHMMSS.log`

---

## Troubleshooting

### Vision camera가 안 열릴 때

* `vision.cam_index`가 올바른지 확인
* `backend`가 시스템에 맞는지 확인 (`v4l2`, `opencv` 등)
* MJPG 설정이 기기와 맞는지 확인 (`mjpg: true/false`)

### YOLO가 너무 민감/둔감할 때

* `yolo_trigger.conf`, `area_thr_ratio`, `hold_frames` 조정

### GRU 트리거가 늦거나 안 걸릴 때

* `seq_len`, `stride`, `ema`, `ready_hold` 조정

### Policy가 로봇을 안 움직일 때

* `policies.policy1.repo_id / policy2.repo_id` 확인
* policy runner에서 `robot.send_action()`까지 action이 만들어지는지 로그로 확인

---