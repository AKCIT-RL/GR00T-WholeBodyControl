"""Joint utility functions and constants for G1 robot.

This module provides joint ordering constants and helper functions for mapping
between motion library data and robot joints.
"""

import torch

# G1 body joint names in IsaacLab order (29 DOF)
G1_ISAACLab_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# G1 hand joint names (14 DOF) - order from g1_43dof.yaml
G1_HAND_JOINTS = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

# Caches for joint indices
_body_joint_indices_cache = {}
_hand_joint_indices_cache = {}


def _get_joint_indices_by_names(asset, joint_names: list, cache: dict) -> torch.Tensor:
    """Get indices of specified joints in the robot's joint list."""
    cache_key = (id(asset), tuple(joint_names))
    if cache_key in cache:
        return cache[cache_key]

    robot_joint_names = asset.joint_names
    indices = [robot_joint_names.index(n) for n in joint_names if n in robot_joint_names]
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=asset.device)
    cache[cache_key] = indices_tensor
    return indices_tensor


# Booster T1 29-DOF body joint names in IsaacLab order (27 actuated DOFs).
# Order derived from URDF BFS traversal (Isaac Lab joint order in simulation):
# [0] Left_Shoulder_Pitch  [1] Right_Shoulder_Pitch  [2] Waist
# [3] Left_Shoulder_Roll   [4] Right_Shoulder_Roll
# [5] Left_Hip_Pitch       [6] Right_Hip_Pitch
# [7] Left_Elbow_Pitch     [8] Right_Elbow_Pitch
# [9] Left_Hip_Roll        [10] Right_Hip_Roll
# [11] Left_Elbow_Yaw      [12] Right_Elbow_Yaw
# [13] Left_Hip_Yaw        [14] Right_Hip_Yaw
# [15] Left_Wrist_Pitch    [16] Right_Wrist_Pitch
# [17] Left_Knee_Pitch     [18] Right_Knee_Pitch
# [19] Left_Wrist_Yaw      [20] Right_Wrist_Yaw
# [21] Left_Ankle_Pitch    [22] Right_Ankle_Pitch
# [23] Left_Hand_Roll      [24] Right_Hand_Roll
# [25] Left_Ankle_Roll     [26] Right_Ankle_Roll
T1_ISAACLab_ORDER = [
    "Left_Shoulder_Pitch",
    "Right_Shoulder_Pitch",
    "Waist",
    "Left_Shoulder_Roll",
    "Right_Shoulder_Roll",
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Left_Elbow_Pitch",
    "Right_Elbow_Pitch",
    "Left_Hip_Roll",
    "Right_Hip_Roll",
    "Left_Elbow_Yaw",
    "Right_Elbow_Yaw",
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Wrist_Pitch",
    "Right_Wrist_Pitch",
    "Left_Knee_Pitch",
    "Right_Knee_Pitch",
    "Left_Wrist_Yaw",
    "Right_Wrist_Yaw",
    "Left_Ankle_Pitch",
    "Right_Ankle_Pitch",
    "Left_Hand_Roll",
    "Right_Hand_Roll",
    "Left_Ankle_Roll",
    "Right_Ankle_Roll",
]

# T1 head joints — extra DOFs not in motion_lib (frozen at default)
T1_HEAD_JOINTS = ["AAHead_yaw", "Head_pitch"]

# Caches for T1
_t1_body_joint_indices_cache = {}
_t1_head_joint_indices_cache = {}


def _is_t1_robot(asset) -> bool:
    """Detect Booster T1 by checking for a T1-specific joint name."""
    return "Left_Shoulder_Pitch" in asset.joint_names


def get_body_joint_indices(asset) -> torch.Tensor:
    """Get indices of body joints using robot-appropriate order.

    Supports G1 (29 DOF) and Booster T1 (27 actuated DOF out of 29 total).
    """
    if _is_t1_robot(asset):
        return _get_joint_indices_by_names(asset, T1_ISAACLab_ORDER, _t1_body_joint_indices_cache)
    return _get_joint_indices_by_names(asset, G1_ISAACLab_ORDER, _body_joint_indices_cache)


def get_hand_joint_indices(asset) -> torch.Tensor:
    """Get indices of extra/hand joints not in the motion library.

    For T1: returns head joint indices (AAHead_yaw, Head_pitch).
    For G1: returns finger joint indices.
    """
    if _is_t1_robot(asset):
        return _get_joint_indices_by_names(asset, T1_HEAD_JOINTS, _t1_head_joint_indices_cache)
    return _get_joint_indices_by_names(asset, G1_HAND_JOINTS, _hand_joint_indices_cache)
