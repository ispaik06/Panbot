
# Panbot 🤖🥞
Vision-triggered robot runtime for **LeRobot SO-ARM101 / SO101 follower arm** with **YOLO segmentation trigger** + **GRU readiness trigger** + **LeRobot pretrained policies (ACT)**.

> **Core idea**
> - One shared **vision trigger camera** (YOLO + GRU) decides **when to switch stages**
> - Separate **policy observation cameras** (LeRobot `robot.cameras`) feed multi-view observations to policies (global/left/right/wrist, etc.)
> - A single runtime orchestrator (`Panbot/control/main_runtime.py`) drives the entire pipeline end-to-end

---

## Demo Video 🎥
Watch the full runtime demo on YouTube:  
👉 [Panbot Demo – Vision-Triggered Runtime (YOLO + GRU + ACT)](https://youtu.be/SyGJ2h8aM98)


---

## Table of Contents
- [What this project does](#what-this-project-does)
- [System architecture](#system-architecture)
- [Vision modules (external repo)](#vision-modules-external-repo)
- [Datasets & pretrained ACT models](#datasets--pretrained-act-models)
- [Folder structure](#folder-structure)
- [Main entry](#main-entry-panbotcontrolmain_runtimepy)
- [Configuration](#configuration-panbotconfigruntimeyaml)
- [Quick start](#quick-start)
- [Runtime stages](#runtime-stages-debug-friendly)
- [Logs](#logs)
- [Troubleshooting](#troubleshooting)
- [Safety notes](#safety-notes)
- [Collaborators](#collaborators)

---

## What this project does

`Panbot/control/main_runtime.py` runs the following sequence:

1. **Connect robot** (SO101 follower arm)
2. **Open vision camera** (single shared camera used by YOLO + GRU triggers)
3. **Run Task 1 motion**, continuously checking the YOLO trigger  
   - When the YOLO trigger fires, the runtime switches Task 1 into a **return sequence**
4. After Task 1 return completes, **wait for GRU trigger**
5. When the GRU trigger fires, **run Policy 1 (Task 2)**
6. Wait for a configured duration, then **run Policy 2 (Task 3)**
7. Finally, **return to base pose + clean/safe shutdown**

### Key distinction (important)
- `vision.cam_index`  
  ✅ One **shared** camera for **YOLO/GRU triggers** (stage switching)
- `robot.cameras`  
  ✅ Separate set of **policy observation cameras** for **LeRobot policies** (multi-view)

---

## System architecture

### Two camera pipelines
**(A) Trigger pipeline (YOLO + GRU)**  
- Input: `vision.cam_index` camera stream  
- Optional: perspective warp using `corners.json`  
- Output: trigger booleans + visualization frames + debug info

**(B) Policy pipeline (LeRobot)**  
- Input: `robot.cameras.*` (right/left/global/wrist, etc.)  
- Used only during policy execution stages  
- Output: actions sent to the robot at `policy_fps`

### Stage switching logic (high-level)
- **YOLO trigger** → signals “Task 1 is done / batter coverage reached” → switch Task 1 into return mode
- **GRU trigger** → signals “cooking readiness reached” → start Task 2 policy (and later Task 3)

---

## Vision modules (external repo)

This runtime repository focuses on **orchestrating** robot control + triggers + policies.

For the vision pipeline implementation details—dataset generation utilities, training code, model/export specifics, and trigger inference modules—please refer to:

- **Panbot_vision:** https://github.com/ispaik06/Panbot_vision

> If you modify vision inference behavior (warp, preprocessing, label mapping, etc.), keep the runtime config (`runtime.yaml`) aligned with the vision repo’s expected preprocessing and checkpoint formats.

---

## Datasets & pretrained ACT models

The runtime expects policy repo IDs in `runtime.yaml` (e.g., `policies.policy1.repo_id`, `policies.policy2.repo_id`).  
Public datasets and pretrained ACT models used for Panbot are available here:

### Hugging Face datasets
- Task 2 dataset: https://huggingface.co/datasets/ispaik06/Panbot_task2_dataset_3  
- Task 3 dataset: https://huggingface.co/datasets/ispaik06/Panbot_task3_dataset_3  

### Hugging Face ACT policy checkpoints
- ACT Task 2: https://huggingface.co/ispaik06/act_panbot_task2_3  
- ACT Task 3: https://huggingface.co/ispaik06/act_panbot_task3_3  

---

## Folder structure

```bash
Panbot/
├─ config/
│  └─ runtime.yaml                  # Runtime config (camera/trigger/tasks/policies/poses/logging)
│
├─ control/
│  └─ main_runtime.py               # ✅ Main runtime orchestrator (end-to-end pipeline)
│
├─ vision/
│  ├─ calibration/
│  │  └─ corners.json               # 4-point corners for perspective warp (optional)
│  │
│  ├─ models/
│  │  └─ runs/
│  │     ├─ batter_seg_local_v1/weights/best.pt     # YOLO segmentation model
│  │     └─ resnet18_gru16_cls/best.pt              # GRU checkpoint
│  │
│  └─ modules/
│     ├─ camera.py                  # open_camera(), resize_for_preview()
│     ├─ yoloseg_infer.py           # YOLOSegConfig, YOLOSegInfer (trigger + visualization)
│     └─ gru_infer.py               # GRUInferConfig, GRUInfer (trigger + visualization)
│
├─ tasks/
│  ├─ base_pose.py                   # BasePoseController, HoldConfig
│  └─ task1_motion.py                # Task1MotionConfig/Stepper, DEFAULT_REST_ACTION
│
├─ policies/
│  └─ common_policy_runner.py        # run_pretrained_policy_shared_robot()
│
└─ logs/
   └─ (runtime logs output here)     # Controlled by runtime.yaml -> log.dir
```

> Paths may vary slightly depending on your local project state, but this README reflects the import paths used by `main_runtime.py`.

---

## Main entry: `Panbot/control/main_runtime.py`

Below is what the runtime imports and how each component is used.

### 1) Runtime config & logging
- Config file: `Panbot/config/runtime.yaml`
- Logging:
  - Reads `log.dir`, `log.level`
  - Logs to both **stdout** and **file**

Related helpers:
- `_load_yaml()`, `_normalize_runtime_config()`
- `_setup_logging(log_dir, level)`

---

### 2) Robot (SO101 follower)
Robot creation:
- `SO101FollowerConfig` from LeRobot
- `make_robot_from_config`

Runtime config keys:
- `robot.port`, `robot.id`, `robot.calibration_dir`
- `robot.cameras.*` (policy observation cameras)

Robot config builder:
- `_build_so101_config(cfg["robot"])`

---

### 3) Vision camera (YOLO/GRU shared)
Vision camera open:
- `from Panbot.vision.modules.camera import open_camera, resize_for_preview`

Runtime config keys:
- `vision.cam_index`, `vision.backend`, `vision.mjpg`
- `vision.width`, `vision.height`, `vision.fps`

Example:
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

### 4) YOLO trigger (segmentation-based)
File:
- `Panbot/vision/modules/yoloseg_infer.py`

Classes:
- `YOLOSegConfig`, `YOLOSegInfer`

Runtime config keys:
- `yolo_trigger.conf`, `yolo_trigger.imgsz`
- `yolo_trigger.area_thr_ratio`, `yolo_trigger.hold_frames`
- `yolo_trigger.use_warp`, `yolo_trigger.warp_w`, `yolo_trigger.warp_h`
- Warp corners file: `paths.corners`

Runtime behavior:
- Call `yolo.step(frame)` → returns `(triggered, vis_frame, info)`
- On first `triggered=True`, Task 1 switches into return mode

---

### 5) GRU trigger (readiness classifier)
File:
- `Panbot/vision/modules/gru_infer.py`

Classes:
- `GRUInferConfig`, `GRUInfer`

Runtime config keys:
- `gru_trigger.image_size`, `gru_trigger.seq_len`, `gru_trigger.stride`
- `gru_trigger.ema`, `gru_trigger.ready_hold`, `gru_trigger.amp`
- `gru_trigger.use_warp`, `gru_trigger.warp_w`, `gru_trigger.warp_h`

Runtime behavior:
- After Task 1 return completes, call `gru.reset()`
- Call `gru.step(frame)` → returns `(triggered, vis_frame, info)`
- When `triggered=True`, start policy stage(s)

---

### 6) Task 1 motion (deterministic motion)
File:
- `Panbot/tasks/task1_motion.py`

Classes:
- `Task1MotionConfig`, `Task1MotionStepper`

Related:
- `DEFAULT_REST_ACTION`

Runtime config keys:
- `task.task1_ramp_time_s`
- `task.task1_pose_hold_s`
- `poses.task1_initial_sequence`
- `poses.task1_return_sequence`

Key calls:
- `task1.start_initial()`
- loop: `task1.step(time.perf_counter())`
- on YOLO trigger: `task1.interrupt_to_return()`

---

### 7) Base pose controller (stability / holding)
File:
- `Panbot/tasks/base_pose.py`

Classes:
- `BasePoseController`, `HoldConfig`

Runtime config keys:
- `poses.base_pose`
- `task.base_pose_hold_interval_s`

Purpose:
- Keeps the robot stable in base pose between stages
- Uses periodic ticks (`base_ctrl.tick()`) to hold position safely

---

### 8) Policies (LeRobot pretrained)
File:
- `Panbot/policies/common_policy_runner.py`

Function:
- `run_pretrained_policy_shared_robot(...)`

Runtime config keys:
- `task.policy_fps`
- `task.task2_duration_s` (Policy 1 duration)
- `task.task3_duration_s` (Policy 2 duration)
- `task.wait_task2_to_task3_s`
- `policies.policy1.repo_id`
- `policies.policy2.repo_id`
- `policies.*.use_amp`, `print_joints`, `print_joints_every`, etc.

Robot actuation happens inside:
- `robot.send_action(...)`

---

## Configuration: `Panbot/config/runtime.yaml`

### Required paths
```yaml
paths:
  corners: "Panbot/vision/calibration/corners.json"
  yolo_model: "Panbot/vision/models/runs/.../best.pt"
  gru_ckpt: "Panbot/vision/models/runs/.../best.pt"
```

### Vision camera (shared trigger camera)
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

Notes:
- `show: true` typically enables local preview windows for debugging.
- `watchdog_s` can be used to detect stalled camera capture (implementation-dependent).

### Robot & policy observation cameras (separate from vision)
```yaml
robot:
  port: "/dev/ttyACM0"
  id: "my_awesome_follower_arm"
  cameras:
    right: { type: "opencv", index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG" }
    # left/global/wrist/... as needed
```

### YOLO trigger parameters
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

### GRU trigger parameters
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

### Task & timing
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

## Quick start

### 1) Install / environment
You need:
- Python 3.x
- LeRobot
- OpenCV
- PyTorch (CUDA recommended)
- YOLO inference dependency (e.g., Ultralytics) if your YOLO module expects it

Example (conceptual):
```bash
conda create -n panbot python=3.10 -y
conda activate panbot

pip install opencv-python torch torchvision
pip install lerobot
pip install ultralytics
```

> CUDA installation depends on your GPU/driver setup. Make sure `torch.cuda.is_available()` is True if you want GPU acceleration.

### 2) Check camera indices
- `vision.cam_index` must point to the **trigger camera**
- `robot.cameras.*.index_or_path` must point to **policy observation cameras**

### 3) Run
```bash
cd ~/Panbot

chmod +x scripts/start_all.sh
./scripts/start_all.sh
```

> If you don’t use `start_all.sh`, you can run the runtime module directly (depends on your package layout):
```bash
python -m Panbot.control.main_runtime
```

---

## Runtime stages (debug-friendly)

- **Stage 1: Task 1 initial + YOLO trigger**
  - `task1.step()` moves the robot
  - `yolo.step(frame)` checks the trigger condition
  - On trigger → switch into Task 1 return sequence

- **Stage 2: Base pose hold + GRU trigger**
  - `base_ctrl.tick()` holds stable base pose
  - `gru.step(frame)` checks readiness trigger

- **Stage 3: Policy 1 (Task 2)**
  - policy runner drives `robot.send_action(...)`

- **Wait**

- **Stage 4: Policy 2 (Task 3)**

- **Finalize**
  - Return to base pose
  - Release camera resources
  - Safe shutdown

---

## Logs
- Location: `log.dir` (default: `Panbot/logs`)
- Filename example: `main_runtime_YYYYMMDD_HHMMSS.log`

---

## Troubleshooting

### Vision camera won’t open
- Verify `vision.cam_index`
- Try different `vision.backend` values (`v4l2`, `opencv`, etc.)
- Toggle `vision.mjpg: true/false` depending on your camera

### YOLO is too sensitive / not sensitive enough
Tune:
- `yolo_trigger.conf`
- `yolo_trigger.area_thr_ratio`
- `yolo_trigger.hold_frames`

### GRU trigger is late or never fires
Tune:
- `gru_trigger.seq_len`, `gru_trigger.stride`
- `gru_trigger.ema`
- `gru_trigger.ready_hold`

Also confirm:
- The GRU checkpoint path is correct (`paths.gru_ckpt`)
- Your preprocessing (warp/resize/normalization) matches training

### Policies don’t move the robot
Check:
- `policies.policy1.repo_id` / `policies.policy2.repo_id`
- Policy runner logs (does it generate actions?)
- Whether `robot.send_action(...)` is reached
- Whether `robot.cameras` are streaming correctly (policy observation inputs)

---

## Safety notes
- Always test with low speed / safe workspace first.
- Keep an emergency stop method available (power cutoff or software kill).
- Make sure your base pose is mechanically safe and collision-free.
- Do not run unattended until the full pipeline is validated.

---

## Collaborators
- **Owner / Main developer:** [ispaik06](https://github.com/ispaik06)
- **Collaborator:** [dongq](https://github.com/dongq)

