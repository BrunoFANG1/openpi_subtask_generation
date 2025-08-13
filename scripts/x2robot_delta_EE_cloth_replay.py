import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device='cuda:5'

import dataclasses
import enum
import logging
import cv2
import socket
import time
import copy
from pathlib import Path
import threading
import sys
import select
import tty
import termios

import tyro
import numpy as np
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from x2robot_dataset.lazy_dataset import (
    X2RDataChunkConfig,
    X2RDataProcessingConfig,
)
from x2robot_dataset.dynamic_robot_dataset import (
    _default_get_frame_fn,
)


import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from x2robot_dataset.common.constants import ACTION_KEY_RANGES

from eval.robot_controller import RobotController

from x2robot_dataset.common.data_utils import (
    convert_euler_to_6D,
    absolute_pose_to_relative_pose,
    relative_pose_to_absolute_pose,
    actions_to_relative,
    relative_to_actions,
)

import numpy as np
from scipy.spatial.transform import Rotation as R
def interpolates_actions(actions, num_actions=20, target_num_actions = 80, action_dim=7):
    # 假设 actions 是你的动作序列，shape 为 [num_actions, action_dim]
    # 其中，欧拉角为 actions[:, 3:6]
    # return interpolated_actions 现在包含了插值后的动作序列，其中角度使用了球面插值
    # 生成目标动作序列的索引
    original_indices = np.linspace(0, num_actions - 1, num_actions)
    target_indices = np.linspace(0, num_actions - 1, target_num_actions)
    # 初始化插值后的动作序列数组
    interpolated_actions = np.zeros((target_num_actions, action_dim))
    # 对[x, y, z, gripper]使用线性插值
    for i in range(3):
        interpolated_actions[:, i] = np.interp(target_indices, original_indices, actions[:, i])
    interpolated_actions[:, -1] = np.interp(target_indices, original_indices, actions[:, -1])
    # 将欧拉角转换为四元数
    quaternions = R.from_euler('xyz', actions[:, 3:6]).as_quat()  # shape: [num_actions, 4]
    # 初始化插值后的四元数数组
    interpolated_quats = np.zeros((target_num_actions, 4))
    # 对四元数进行球面插值
    for i in range(4):  # 对四元数的每个分量进行插值
        interpolated_quats[:, i] = np.interp(target_indices, original_indices, quaternions[:, i])
    # 四元数规范化，确保插值后仍为单位四元数
    interpolated_quats = interpolated_quats / np.linalg.norm(interpolated_quats, axis=1, keepdims=True)
    # 将插值后的四元数转换回欧拉角
    interpolated_eulers = R.from_quat(interpolated_quats).as_euler('xyz')  # shape: [target_num_actions, 3]
    # 更新插值后动作序列的角度部分
    interpolated_actions[:, 3:6] = interpolated_eulers
    # print(interpolated_actions.shape)
    return interpolated_actions

MODEL_TO_ROBOT_MAPPING = {
    # Arm cartesian pose (7D: 3 pos, 3 rot, 1 gripper)
    'master_left_ee_cartesian_pos':  {'name': 'follow1_pos', 'shape': 3, 'slice': slice(0, 3)},
    'master_left_ee_cartesian_pos_relative': {'name': 'follow1_pos', 'shape': 3, 'slice': slice(0, 3)},
    'follow_left_ee_cartesian_pos':  {'name': 'follow1_pos', 'shape': 3, 'slice': slice(0, 3)},
    'master_left_ee_rotation':       {'name': 'follow1_pos', 'shape': 3, 'slice': slice(3, 6)},
    'master_left_ee_rotation_relative': {'name': 'follow1_pos', 'shape': 3, 'slice': slice(3, 6)},
    'follow_left_ee_rotation':       {'name': 'follow1_pos', 'shape': 3, 'slice': slice(3, 6)},
    'follow_left_gripper':           {'name': 'follow1_pos', 'shape': 1, 'slice': slice(6, 7)},
    'master_right_ee_cartesian_pos': {'name': 'follow2_pos', 'shape': 3, 'slice': slice(0, 3)},
    'master_right_ee_cartesian_pos_relative': {'name': 'follow2_pos', 'shape': 3, 'slice': slice(0, 3)},
    'follow_right_ee_cartesian_pos': {'name': 'follow2_pos', 'shape': 3, 'slice': slice(0, 3)},
    'master_right_ee_rotation':      {'name': 'follow2_pos', 'shape': 3, 'slice': slice(3, 6)},
    'master_right_ee_rotation_relative': {'name': 'follow2_pos', 'shape': 3, 'slice': slice(3, 6)},
    'follow_right_ee_rotation':      {'name': 'follow2_pos', 'shape': 3, 'slice': slice(3, 6)},
    'follow_right_gripper':          {'name': 'follow2_pos', 'shape': 1, 'slice': slice(6, 7)},
    
    # Gripper current (from robot's followX_joints_cur)
    'follow_left_gripper_cur':       {'name': 'follow1_joints_cur', 'shape': 1, 'slice': slice(-1, None)},
    'follow_right_gripper_cur':      {'name': 'follow2_joints_cur', 'shape': 1, 'slice': slice(-1, None)},

    # Other modalities
    'velocity_decomposed':           {'name': 'car_pose', 'shape': 3},
    'height':                        {'name': 'lift', 'shape': 1},
    'head_rotation':                 {'name': 'head_pos', 'shape': 2},
}

def normalize_action(action_input, min_range, max_range, mean=None, std=None):
    if mean is None:
        return (action_input - min_range) / (max_range - min_range)
    else:
        return (action_input - mean) / std

def unnormalize_action(action_input, min_range, max_range, mean=None, std=None):
    if mean is None:
        return action_input * (max_range - min_range) + min_range
    else:
        return action_input * std + mean

def velocity_to_pose(vx_body, vy_body, vyaw, dt, start_pose):
    """Converts body frame velocity to global frame pose."""
    x, y, theta = start_pose
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    dx_global = (vx_body * cos_theta - vy_body * sin_theta) * dt
    dy_global = (vx_body * sin_theta + vy_body * cos_theta) * dt
    dtheta = vyaw * dt
    x_new, y_new = x + dx_global, y + dy_global
    theta_new = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi
    return np.array([x_new, y_new, theta_new])

def build_action_masks(pred_keys, mapping):
    """Builds boolean masks for slicing the flat action tensor."""
    total_dim = sum(mapping.get(k, {'shape': 0})['shape'] for k in pred_keys)
    masks, cursor = {}, 0
    for k in pred_keys:
        if k not in mapping: continue
        dim = mapping[k]['shape']
        m = np.zeros(total_dim, dtype=bool)
        m[cursor:cursor + dim] = True
        masks[k] = m
        cursor += dim
    return masks

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    X2ROBOT = "x2robot"

@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str

@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    log_replay: bool = False
    openloop: bool = False
    action_horizon: int = 20 # prefix_attention_horizon + execution_horizon
    inference_delay: int = 4
    prefix_attention_horizon: int = 12
    execution_horizon: int = 8
    prefix_attention_schedule: str = 'exp'
    max_guidance_weight: list = dataclasses.field(default_factory=lambda: [5.0])
    custom_norm_stats_path: None = None


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")

def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)

def main(args: Args) -> None:

    # Load Dataset
    action_horizon = 20
    episode_item = {
        "path": '/x2robot_data/zhengwei/10053/20250731-day-sort_and_fold_cloth_high_quality/20250731-day-sort_and_fold_cloth_high_quality@MASTER_SLAVE_MODE@2025_07_31_14_21_08',
        "st_frame": 50,
        "ed_frame": 1200,
    }

    agent_pos_keys = [
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation",
        "follow_left_gripper",
        "follow_right_ee_cartesian_pos",
        "follow_right_ee_rotation",
        "follow_right_gripper",
        "follow_left_gripper_cur",
        "follow_right_gripper_cur",
    ]
    prediction_action_keys = [
        "master_left_ee_cartesian_pos_relative",
        "master_left_ee_rotation_relative",
        "follow_left_gripper",
        "master_right_ee_cartesian_pos_relative",
        "master_right_ee_rotation_relative",
        "follow_right_gripper",
    ]
    data_config = X2RDataProcessingConfig()
    from x2robot_dataset.common.data_preprocessing import _CAM_MAPPING
    cam_mapping = _CAM_MAPPING
    data_config.update(
        cam_mapping=cam_mapping,
        class_type="x2",
        train_test_split=0.9,
        filter_angle_outliers=False,
        sample_rate=1,
        parse_tactile=False,
        obs_action_keys=agent_pos_keys,
        predict_action_keys=prediction_action_keys,
        trim_stationary=False,
        one_by_one_relative=False,
        distributed_instruction_ratio=0.0,
        cache_dir = "/x2robot_v2/xinyuanfang/projects_v2/x2robot_dataset"
    )
    norm_stats = {}
    predict_action_min, predict_action_max, agent_pos_min, agent_pos_max = [], [], [], []
    for key in data_config.predict_action_keys:
        predict_action_min += ACTION_KEY_RANGES[key]['min_range']
        predict_action_max += ACTION_KEY_RANGES[key]['max_range']
    for key in data_config.obs_action_keys:
        agent_pos_min += ACTION_KEY_RANGES[key]['min_range']
        agent_pos_max += ACTION_KEY_RANGES[key]['max_range']
    # TODO: Temp fix
    if args.custom_norm_stats_path is not None:
        with open(args.custom_norm_stats_path, 'r') as f:
            import json
            custom_norm_stats = json.load(f)
        norm_stats['action_mean'] = np.array(custom_norm_stats['norm_stats']['action']['mean'])
        norm_stats['action_std'] = np.array(custom_norm_stats['norm_stats']['action']['std'])
        norm_stats['state_mean'] = np.array(custom_norm_stats['norm_stats']['agent_pos']['mean'])
        norm_stats['state_std'] = np.array(custom_norm_stats['norm_stats']['agent_pos']['std'])
        print(f"Using custom normalization stats from {args.custom_norm_stats_path}")
    else:
        norm_stats = None
    data_config.update(
        norm_stats=norm_stats,
    )
    data_chunk_config = X2RDataChunkConfig().update(
        left_padding=False,
        right_padding=True,
        action_horizon=action_horizon,
        action_history_length=0,
        predict_action_keys=prediction_action_keys,
    )
    print(data_chunk_config.predict_action_keys)
    print(data_chunk_config.use_relative_action)
    # check_episode_fn = _default_check_episode_fn
    get_frame_fn = _default_get_frame_fn
    frames = get_frame_fn(episode_item, data_config, data_chunk_config)

    # Load Policy Model
    policy = create_policy(args)

    ground_truth, model_output = [], []
    count = 0
    while True:
        # print(len(frames))
        camera_left = frames[count]['obs']['left_wrist_view'][0].reshape(480,640,3)
        camera_front = frames[count]['obs']['face_view'][0].reshape(480,640,3)
        camera_right = frames[count]['obs']['right_wrist_view'][0].reshape(480,640,3)
        state = frames[count]['obs']['agent_pos'][0]
        norm_state = normalize_action(np.array(state), np.array(agent_pos_min), np.array(agent_pos_max))
        obs = {
            'images': {
                'left_wrist_view': camera_left,
                'face_view': camera_front,
                'right_wrist_view': camera_right,
            },
            'prompt': 'sort and fold cloth',
            'state': norm_state,
        }
        result = policy.infer(obs)
        action_pred = result['actions']
        unnormalized_action = unnormalize_action(np.array(action_pred), np.array(predict_action_min), np.array(predict_action_max))
        # unnormalized_label = unnormalize_action(frames[count]['action'], predict_action_min, predict_action_max, norm_stats['action_mean'], norm_stats['action_std'])
        # import pdb; pdb.set_trace()
        # pred_action = relative_to_actions(unnormalized_action, state[0:14])

        ground_truth.append(frames[count]['action'])
        model_output.append(unnormalized_action)
        
        count += action_horizon
        if count >= len(frames)-1:
            break

    # Generate plot
    ground_truth = np.stack(ground_truth, axis=0)
    model_output = np.stack(model_output, axis=0)

    # Insert np.nan between chunks to create discontinuities in the plot
    num_chunks, action_horizon, dim = ground_truth.shape
    nan_separator = np.full((num_chunks, 1, dim), np.nan)
    ground_truth_gapped = np.concatenate((ground_truth, nan_separator), axis=1).reshape(-1, dim)
    model_output_gapped = np.concatenate((model_output, nan_separator), axis=1).reshape(-1, dim)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4 * dim))

    for i in range(dim):
        plt.subplot(dim, 1, i + 1)

        # Set x-ticks to correspond to the start of each action chunk
        tick_locations = [k * (action_horizon + 1) for k in range(num_chunks)]
        tick_labels = [k * action_horizon for k in range(num_chunks)]
        plt.xticks(tick_locations, tick_labels, rotation=45)
        plt.grid(axis='x', linestyle='--')

        plt.plot(ground_truth_gapped[:, i], label='Ground Truth', color='blue')
        plt.plot(model_output_gapped[:, i], label='Model Output', color='orange')
        plt.title(f'Action Dimension {i + 1}')
        plt.xlabel('Time Step')
        plt.ylabel('Action Value')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('x2robot_delta_EE_cloth_replay.png')
    print(f"plot saved to x2robot_delta_EE_cloth_replay.png")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))