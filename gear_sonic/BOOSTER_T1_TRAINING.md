# SONIC on Booster T1 (29-DOF) — Porting Guide

This document describes how the NVIDIA GR00T SONIC whole-body motion-imitation policy was ported from the **Unitree G1** to the **Booster T1** (29-DOF variant, `nu=27`). It covers every file added or modified, key technical decisions, validated results, and the remaining steps to reach full-scale training.

> **Branch:** `t1_sonic`  
> **Base policy:** SONIC (this repo, `gear_sonic/`)  
> **Retargeting:** [AKCIT-RL/GMR](https://github.com/AKCIT-RL/GMR) — branch `pr/169`  
> **Motion dataset:** [Bones-SEED](https://huggingface.co/datasets/bones-studio/seed) (BVH subset, 53 files)  
> **Validated on:** Isaac Lab 0.54.3 / Isaac Sim 5.1, RTX 4090 24 GB

---

## Table of Contents

1. [Robot Facts](#1-robot-facts)
2. [Files Added / Modified](#2-files-added--modified)
3. [Step-by-Step Reproduction](#3-step-by-step-reproduction)
4. [Key Technical Decisions](#4-key-technical-decisions)
5. [Validation Results](#5-validation-results)
6. [Training Commands](#6-training-commands)
7. [Evaluation](#7-evaluation)
8. [Next Steps](#8-next-steps)

---

## 1. Robot Facts

| Property | Value |
|---|---|
| Robot | Booster T1 (`booster_t1_29dof`) |
| nu (actuated DOFs) | **27** (the "29dof" name is a GMR convention counting 2 unactuated head joints) |
| nq | 34 (7 free-joint + 27 revolute) |
| nbody (MJCF) | 33 (world + 32 links) |
| nbody (Isaac Lab) | 30 (URDF links, floating base merged) |
| Head joints | `AAHead_yaw`, `Head_pitch` — present in URDF but **not** in motion_lib |

### MuJoCo actuator order (= DOF order in PKLs)

| Index | Joint | Torque (Nm) |
|---|---|---|
| 0–6 | Left arm: Shoulder_Pitch/Roll, Elbow_Pitch/Yaw, Wrist_Pitch/Yaw, Hand_Roll | ±18 |
| 7–13 | Right arm (same structure) | ±18 |
| 14 | Waist | ±30 |
| 15–20 | Left leg: Hip_Pitch/Roll/Yaw, Knee_Pitch, Ankle_Pitch/Roll | ±45/±24/±15 |
| 21–26 | Right leg (same structure) | ±45/±24/±15 |

### Isaac Lab BFS joint order (from URDF traversal — **differs** from MJCF)

Isaac Lab traverses the URDF in BFS order, producing an **interleaved** left/right ordering that does **not** match the MJCF's grouped (all-left-arm, all-right-arm) layout. The full 29-joint BFS list (all joints including head):

```
[0]  AAHead_yaw          [1]  Left_Shoulder_Pitch   [2]  Right_Shoulder_Pitch
[3]  Waist               [4]  Head_pitch            [5]  Left_Shoulder_Roll
[6]  Right_Shoulder_Roll [7]  Left_Hip_Pitch        [8]  Right_Hip_Pitch
[9]  Left_Elbow_Pitch    [10] Right_Elbow_Pitch     [11] Left_Hip_Roll
[12] Right_Hip_Roll      [13] Left_Elbow_Yaw        [14] Right_Elbow_Yaw
[15] Left_Hip_Yaw        [16] Right_Hip_Yaw         [17] Left_Wrist_Pitch
[18] Right_Wrist_Pitch   [19] Left_Knee_Pitch       [20] Right_Knee_Pitch
[21] Left_Wrist_Yaw      [22] Right_Wrist_Yaw       [23] Left_Ankle_Pitch
[24] Right_Ankle_Pitch   [25] Left_Hand_Roll        [26] Right_Hand_Roll
[27] Left_Ankle_Roll     [28] Right_Ankle_Roll
```

The derived **DOF permutation** (head joints 0, 4 excluded → 27-element arrays):

```python
T1_ISAACLAB_TO_MUJOCO_DOF = [
    0, 7, 14, 1, 8, 15, 21, 2, 9, 16, 22, 3, 10, 17, 23,
    4, 11, 18, 24, 5, 12, 19, 25, 6, 13, 20, 26,
]
T1_MUJOCO_TO_ISAACLAB_DOF = [
    0, 3, 7, 11, 15, 19, 23, 1, 4, 8, 12, 16, 20, 24, 2,
    5, 9, 13, 17, 21, 25, 6, 10, 14, 18, 22, 26,
]
```

---

## 2. Files Added / Modified

> G1 and H2 files are **untouched**. All T1 changes are additive.

### New files

| File | Purpose |
|---|---|
| `gear_sonic/envs/manager_env/robots/booster_t1_29dof.py` | Robot config: joints, actuators, KP/KD, DOF/body mappings, action scales |
| `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_t1.yaml` | Hydra experiment config (SOMA encoder mode, all T1 body-name overrides) |
| `gear_sonic/data/assets/robot_description/urdf/booster_t1_29dof/` | 29-DOF URDF + 52 STL meshes for Isaac Lab |
| `gear_sonic/data/assets/robot_description/mjcf/booster_t1_29dof.xml` | MuJoCo XML for motion-library FK |
| `external_dependencies/GMR` | Submodule — [AKCIT-RL/GMR](https://github.com/AKCIT-RL/GMR) at `pr/169` |
| `external_dependencies/htwk-gym` | Submodule — [NaoHTWK/htwk-gym](https://github.com/NaoHTWK/htwk-gym) (base URDF source) |

### Modified files

| File | What changed |
|---|---|
| `gear_sonic/envs/manager_env/robots/__init__.py` | Added `from gear_sonic.envs.manager_env.robots.booster_t1_29dof import *` |
| `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` | Added `"booster_t1_29dof"` entry to `robot_mapping` dict |
| `gear_sonic/trl/utils/order_converter.py` | Added `T1Converter` class |
| `gear_sonic/envs/env_utils/joint_utils.py` | Added T1 joint-order arrays and auto-detection for the DOF-mismatch code path |

---

## 3. Step-by-Step Reproduction

### Prerequisites

```bash
# Python envs
# Offline preprocessing (retargeting, PKL conversion):
python -m venv GMR/.venv && source GMR/.venv/bin/activate && pip install -e GMR/

# Isaac Lab training:
conda activate env_isaaclab
pip install -e "GR00T-WholeBodyControl/gear_sonic/[training]"
```

### A — Verify robot model

```bash
python -c "
import mujoco as mj
m = mj.MjModel.from_xml_path(
    'GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/booster_t1_29dof.xml')
print(f'nbody={m.nbody}  nu={m.nu}  nq={m.nq}')
# Expected: nbody=33  nu=27  nq=34
"
```

### B — Retarget BVH motions to T1 (GMR)

```bash
source GMR/.venv/bin/activate
cd GMR

python scripts/bvh_to_robot_dataset.py \
  --src_folder ../data/bones_seed_100/bvh \
  --tgt_folder ../data/t1_motions_raw \
  --config assets/booster_t1_29dof/ik_config_t1.json \
  --fps 120
```

### C — Convert to SONIC motion_lib format

```bash
python scripts/04_convert_to_motion_lib.py \
  --input ../data/t1_motions_raw \
  --output ../data/t1_motion_lib \
  --mirror        # generates _M.pkl mirrored variants
```

### D — Extract SOMA data from BVH

```bash
source GMR/.venv/bin/activate
cd GR00T-WholeBodyControl

python gear_sonic/data_process/extract_soma_joints_from_bvh.py \
  --input ../data/bones_seed_100/bvh \
  --output ../data/t1_soma \
  --fps 30 --num_workers 8
```

### E — Smoke test (MuJoCo, no Isaac Lab needed)

```bash
# Kinematic replay of one motion
python -c "
import pickle, mujoco, mujoco.viewer, numpy as np

with open('data/t1_motion_lib/dancecards1_AB_normal_001__A005.pkl','rb') as f:
    d = list(pickle.load(f).values())[0]

m = mujoco.MjModel.from_xml_path(
    'GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/booster_t1_29dof.xml')
data = mujoco.MjData(m)
print('frames:', d['dof'].shape[0], '  DOFs:', d['dof'].shape[1])
"
```

### F — Isaac Lab smoke test (2 iterations)

```bash
conda activate env_isaaclab
cd GR00T-WholeBodyControl

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_t1 \
  num_envs=1 headless=True use_wandb=false \
  algo.config.num_learning_iterations=2 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/path/to/data/t1_motion_lib \
  ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=/path/to/data/t1_soma
```

Expected: two training iterations complete without error.

---

## 4. Key Technical Decisions

### 4.1 DOF mismatch: motion_lib (27) vs Isaac Lab (29)

The URDF has 29 revolute joints (27 body + 2 head). The motion_lib PKLs have 27 DOFs. This triggers the SONIC "DOF mismatch" code path. To enable it:

```yaml
# sonic_t1.yaml
manager_env:
  commands:
    motion:
      motion_lib_num_dof: 27   # < robot_num_dof (29) → activates mismatch path
```

`gear_sonic/envs/env_utils/joint_utils.py` was extended with:

```python
T1_ISAACLab_ORDER = [
    "Left_Shoulder_Pitch", "Right_Shoulder_Pitch", "Waist",
    "Left_Shoulder_Roll", "Right_Shoulder_Roll",
    # ... 27 joints in Isaac Lab BFS order, excluding head
]
T1_HEAD_JOINTS = ["AAHead_yaw", "Head_pitch"]  # excluded from motion_lib
```

The `get_body_joint_indices()` and `get_hand_joint_indices()` functions auto-detect T1 by checking for `"Left_Shoulder_Pitch"` in `asset.joint_names`.

### 4.2 Isaac Lab BFS ≠ MJCF declaration order

Using an identity DOF mapping silently scrambles all observations and actions. The correct permutation was derived from the joint list printed in Isaac Lab error messages during initial `num_envs=1` runs. See the arrays in section 1.

### 4.3 SOMA data at 30 fps, robot PKLs at 120 fps

BVH source files are 120 fps. GMR outputs at source rate (120 fps). The SOMA extraction pipeline runs at 30 fps. All robot PKLs must be downsampled to 30 fps (or SOMA upsampled) so the motion-library frame counts match. The PKLs in `data/t1_motion_lib/` are stored at 30 fps.

### 4.4 PKL schema: `{motion_name: data_dict}`

SONIC's motion_lib lazy-loading (directory mode) expects each PKL to be a `dict` keyed by motion name:

```python
{"dancecards1_AB_normal_001__A005": {"root_trans_offset": ..., "dof": ..., ...}}
```

Raw GMR output is a flat `dict` (not wrapped). The conversion script wraps each file.

### 4.5 SOMA encoder (not SMPL)

`sonic_t1.yaml` uses the SOMA encoder (`unitoken_all_noz_soma`, `all_mlp_v1_soma`) rather than the SMPL encoder, because:
- SOMA 26-joint data can be extracted directly from BVH files via `extract_soma_joints_from_bvh.py`
- SMPL data would require an additional SMPL fitting step per BVH

Set `smpl_motion_file: dummy` to disable the SMPL encoder.

---

## 5. Validation Results

| Test | Config | Result |
|---|---|---|
| MJCF loads in MuJoCo | `mujoco.MjModel.from_xml_path(...)` | ✅ `nbody=33 nu=27 nq=34` |
| URDF loads in MuJoCo | `mujoco.MjModel.from_xml_path(...)` | ✅ `nbody=30 nu=0 nq=36` |
| Kinematic replay (EGL) | 10 motions, 30 fps render | ✅ 10/10, 10 MP4s |
| Physics kinematic | FK + joint limits + foot_z | ✅ 10/10 |
| Isaac Lab smoke (`num_envs=1`, 2 iter) | `sonic_t1.yaml` | ✅ No error |
| Isaac Lab SOMA (`num_envs=1`, 2 iter) | SOMA encoder active | ✅ No error |
| Short training (`num_envs=16`, 995 iter) | 20 PKLs (10 + 10 mirrored) | ✅ Stable, no crash |
| Overfit sanity (`num_envs=8`, 5000 iter, 1 motion) | `dancecards1_AB_normal_001` | 🔄 In progress |

---

## 6. Training Commands

### Overfit sanity test (single motion, low VRAM)

```bash
conda activate env_isaaclab
cd GR00T-WholeBodyControl

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_t1 \
  num_envs=8 headless=True use_wandb=false \
  algo.config.num_learning_iterations=5000 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/path/to/data/t1_motion_lib/dancecards1_AB_normal_001__A005.pkl \
  ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=/path/to/data/t1_soma/dancecards1_AB_normal_001__A005.pkl
```

**Sanity criterion:** mean reward should rise from ~−0.05 to > 0 within 500 iterations. If it stays flat, check the DOF mapping and body-name overrides.

### Short training run (16 envs)

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_t1 \
  num_envs=16 headless=True use_wandb=false \
  algo.config.num_learning_iterations=1000 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/path/to/data/t1_motion_lib \
  ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=/path/to/data/t1_soma
```

### Full training (F4)

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_t1 \
  num_envs=4096 headless=True use_wandb=true \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/path/to/data/t1_motion_lib \
  ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=/path/to/data/t1_soma
```

Requires ~40 GB VRAM (multi-GPU or A100/H100).

---

## 7. Evaluation

Load a trained checkpoint and visualise in the Isaac Lab viewer:

```bash
conda activate env_isaaclab
cd GR00T-WholeBodyControl

python gear_sonic/eval_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_t1 \
  headless=False num_envs=1 \
  checkpoint=logs_rl/TRL_T1_Track/<run_dir>/model_XXXX.pt \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/path/to/data/t1_motion_lib \
  ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=/path/to/data/t1_soma
```

Checkpoints are saved to `logs_rl/TRL_T1_Track/<experiment_name>-<timestamp>/`.

---

## 8. Next Steps

- [ ] **Evaluate overfit run** — inspect reward curve; visual check with `eval_agent_trl.py headless=False`
- [ ] **KP/KD tuning** — current gains estimated from torque limits; refine if robot oscillates
- [ ] **Expand motion dataset** — retarget remaining 43 BVH files; target ≥200 diverse motions
- [ ] **Full training (F4)** — `num_envs=4096`, multi-GPU, `use_wandb=true`
- [ ] **Verify `T1_ISAACLAB_TO_MUJOCO_BODY`** — currently identity; cross-check FK poses in both sims
- [ ] **`sim2mujoco` T1 port** — add T1 MJCF/config to `decoupled_wbc/sim2mujoco/` for lightweight eval
- [ ] **Real-robot deployment** — export to ONNX via `gear_sonic_deploy/` and test on physical T1
- [ ] **SMPL data** — download from `nvidia/GEAR-SONIC` to replace `smpl_motion_file: dummy`
- [ ] **Mirrored motions for all 53 BVHs** — currently only 10 retargeted files have `_M.pkl` variants
