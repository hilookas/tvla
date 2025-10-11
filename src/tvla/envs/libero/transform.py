import numpy as np

from robosuite.utils.transform_utils import quat2mat, mat2quat, quat2axisangle, axisangle2quat
from robosuite.utils.control_utils import set_goal_orientation, set_goal_position
from pytransform3d.transformations import invert_transform
from pytransform3d.rotations import quaternion_from_matrix, matrix_from_quaternion

# Transformation of Unified Explicit Action Representation (UEA) to LIBERO
        
# Transformation matrix from TCP (Tool Center Point) to end-effector frame
Ttcp2eef = np.array([
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
])


def uea2libero_state(
    Ttcp2cam, gripper_open, 
    Tcam2world
):
    # Restore end-effector pose in world frame from last observation
    Teef2world = Tcam2world @ Ttcp2cam @ invert_transform(Ttcp2eef)
    
    # Restore EEF
    eef_pos = Teef2world[:3, 3]
    # eef_quat = mat2quat(Teef2world[:3, :3])
    # For quaternions, -q has the same physical meaning as q, 
    # but here we need to ensure consistency with the untransformed representation; 
    # i.e., mat2quat(quat2mat(q)) should return the same q.
    eef_quat = np.roll(quaternion_from_matrix(Teef2world[:3, :3]), -1)
    # eef_axis_angle = quat2axisangle(eef_quat)
    
    # Restore gripper
    gripper_qpos = np.array([gripper_open / 2, -gripper_open / 2])
    
    return eef_pos, eef_quat, gripper_qpos


def libero2uea_state(
    eef_pos, eef_quat, gripper_qpos, 
    Tcam2world
):
    # Construct end-effector pose in world frame
    Teef2world = np.eye(4)
    Teef2world[:3, 3] = eef_pos  # Position
    # Teef2world[:3, :3] = quat2mat(eef_quat)  # Orientation
    Teef2world[:3, :3] = matrix_from_quaternion(np.roll(eef_quat, 1))  # Orientation

    # Compute TCP pose in camera frame
    Ttcp2cam = invert_transform(Tcam2world) @ Teef2world @ Ttcp2eef
    
    # Compute gripper opening
    gripper_open = float(gripper_qpos[0] - gripper_qpos[1])
    
    return Ttcp2cam, gripper_open


def uea2libero_action(
    Ttcp2cam, gripper_open, 
    last_obs_Ttcp2cam, Tcam2world
):
    # Restore end-effector pose in world frame from last observation
    Teefstate2world = Tcam2world @ last_obs_Ttcp2cam @ invert_transform(Ttcp2eef)
    
    # Compute target end-effector pose in world frame from action
    Teefaction2world = Tcam2world @ Ttcp2cam @ invert_transform(Ttcp2eef)  # Absolute control
    
    # Compute position delta in libero way
    delta_pos = Teefaction2world[:3, 3] - Teefstate2world[:3, 3]
    
    # Compute orientation delta in libero way
    # action_ori = rotation_mat_error @ obs_eef_ori_mat
    # So rotation_mat_error = action_ori @ obs_eef_ori_mat.T
    rotation_mat_error = Teefaction2world[:3, :3] @ Teefstate2world[:3, :3].T
    # delta_quat = mat2quat(rotation_mat_error)
    delta_quat = np.roll(quaternion_from_matrix(rotation_mat_error), -1)
    delta_axis_angle = quat2axisangle(delta_quat)
    
    # Restore scaling factors in libero way
    scaled_delta_pos = delta_pos * 20
    scaled_delta_ori = delta_axis_angle * 2
    
    # Restore gripper action
    # action_gripper_qpos = (-gripper_action + 1) * 0.04
    # So gripper_action = 1 - action_gripper_qpos / 0.04
    gripper_action = 1 - gripper_open / 0.04
    
    return scaled_delta_pos, scaled_delta_ori, gripper_action


def libero2uea_action(
    scaled_delta_pos, scaled_delta_ori, gripper_action, 
    last_obs_Ttcp2cam, Tcam2world
):
    # Restore end-effector pose in world frame from last observation
    Teefstate2world = Tcam2world @ last_obs_Ttcp2cam @ invert_transform(Ttcp2eef)
    
    # scale actions:
    # /root/miniforge3/envs/libero/lib/python3.8/site-packages/robosuite/controllers/base_controller.py
    scaled_delta_pos = scaled_delta_pos / 20
    scaled_delta_ori = scaled_delta_ori / 2
    
    goal_ori = set_goal_orientation(scaled_delta_ori, Teefstate2world[:3, :3])
    goal_pos = set_goal_position(scaled_delta_pos, Teefstate2world[:3, 3])
    
    Teefaction2world = np.eye(4)
    Teefaction2world[:3, :3] = goal_ori
    Teefaction2world[:3, 3] = goal_pos

    # Compute TCP pose in camera frame
    Ttcp2cam = invert_transform(Tcam2world) @ Teefaction2world @ Ttcp2eef
    
    # only two type of action: 1 stand for closed, -1 stand for open, 0.04 is max open 
    # /root/miniforge3/envs/libero/lib/python3.8/site-packages/robosuite/models/grippers/panda_gripper.py
    gripper_open = (-gripper_action + 1) * 0.04
    
    return Ttcp2cam, gripper_open