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
        "st_frame": 100,
        "ed_frame": 200,
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
    agent_pos_min = np.array(agent_pos_min)
    agent_pos_max = np.array(agent_pos_max)
    predict_action_min = np.array(predict_action_min)
    predict_action_max = np.array(predict_action_max)
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

    model_outputs, rtc_outputs, prev_chunks = [], [], []
    count = 0
    
    # Generate the first chunk for rtc inference
    camera_left = frames[count]['obs']['left_wrist_view'][0].reshape(480,640,3)
    camera_front = frames[count]['obs']['face_view'][0].reshape(480,640,3)
    camera_right = frames[count]['obs']['right_wrist_view'][0].reshape(480,640,3)
    state = frames[count]['obs']['agent_pos'][0]
    norm_state = normalize_action(state, agent_pos_min, agent_pos_max)
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
    unnormalized_action = unnormalize_action(action_pred, predict_action_min, predict_action_max)
    
    # Warning: This is for delta EE pose 
    absEE = relative_to_actions(unnormalized_action, state[0:14])
    next_inference_start_EE_traj = absEE[args.execution_horizon-1 :]
    next_inference_start_EE_traj[0] = frames[count+args.execution_horizon-1]['obs']['agent_pos'][0][0:14]
    prev_relative_action = actions_to_relative(next_inference_start_EE_traj)[1:]
    assert prev_relative_action.shape == (args.action_horizon - args.execution_horizon, 14)
    prev_chunk = np.concatenate([
            prev_relative_action, 
            np.zeros((args.execution_horizon, 14)),
        ]) 
    count += args.execution_horizon

    # Main loop
    while True:
        camera_left = frames[count]['obs']['left_wrist_view'][0].reshape(480,640,3)
        camera_front = frames[count]['obs']['face_view'][0].reshape(480,640,3)
        camera_right = frames[count]['obs']['right_wrist_view'][0].reshape(480,640,3)
        state = frames[count]['obs']['agent_pos'][0]
        norm_state = normalize_action(state, agent_pos_min, agent_pos_max)
        norm_prev_chunk = normalize_action(prev_chunk, predict_action_min, predict_action_max)
        obs = {
            'images': {
                'left_wrist_view': camera_left,
                'face_view': camera_front,
                'right_wrist_view': camera_right,
            },
            'prompt': 'sort and fold cloth',
            'state': norm_state,
        }
        rtc_result = policy.infer_rtc(obs, norm_prev_chunk, args.inference_delay, args.prefix_attention_horizon, args.max_guidance_weight[0])
        rtc_action_pred = rtc_result['actions']
        raw_result = policy.infer(obs)
        raw_action_pred = raw_result['actions']
        unnormalized_raw_action = unnormalize_action(raw_action_pred, predict_action_min, predict_action_max)
        unnormalized_rtc_action = unnormalize_action(rtc_action_pred, predict_action_min, predict_action_max)
        prev_chunks.append(prev_chunk)
        model_outputs.append(unnormalized_raw_action)
        rtc_outputs.append(unnormalized_rtc_action)
        
        # Update prev chunk
        # Warning: This is for delta EE pose 
        absEE = relative_to_actions(unnormalized_raw_action, state[0:14])
        next_inference_start_EE_traj = absEE[args.execution_horizon-1 :]
        next_inference_start_EE_traj[0] = frames[count+args.execution_horizon-1]['obs']['agent_pos'][0][0:14]
        prev_relative_action = actions_to_relative(next_inference_start_EE_traj)[1:]
        assert prev_relative_action.shape == (args.action_horizon - args.execution_horizon, 14)
        prev_chunk = np.concatenate([
                prev_relative_action, 
                np.zeros((args.execution_horizon, 14)),
            ]) 
        count += args.execution_horizon
        if count >= len(frames)-1-args.action_horizon:
            break

    # Generate plot
    model_output = np.stack(model_outputs, axis=0)
    rtc_output = np.stack(rtc_outputs, axis=0)
    prev_chunk = np.stack(prev_chunks, axis=0)
    prev_chunk[:, args.prefix_attention_horizon :, :] = np.nan

    # Insert np.nan between chunks to create discontinuities in the plot
    num_chunks, action_horizon, dim = model_output.shape
    nan_separator = np.full((num_chunks, 1, dim), np.nan)
    model_output_gapped = np.concatenate((model_output, nan_separator), axis=1).reshape(-1, dim)
    prev_chunk_gapped = np.concatenate((prev_chunk, nan_separator), axis=1).reshape(-1, dim)
    rtc_output_gapped = np.concatenate((rtc_output, nan_separator), axis=1).reshape(-1, dim)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4 * dim))

    for i in range(dim):
        plt.subplot(dim, 1, i + 1)

        # Set x-ticks to correspond to the start of each action chunk
        tick_locations = [k * (action_horizon + 1) for k in range(num_chunks)]
        tick_labels = [k * action_horizon for k in range(num_chunks)]
        plt.xticks(tick_locations, tick_labels, rotation=45)
        plt.grid(axis='x', linestyle='--')

        plt.plot(model_output_gapped[:, i], label='Raw Output', color='orange')
        plt.plot(rtc_output_gapped[:, i], label='RTC Output', color='green')
        plt.plot(prev_chunk_gapped[:, i], label='Prev Chunk', color='blue')
        plt.title(f'Action Dimension {i + 1}')
        plt.xlabel('Time Step')
        plt.ylabel('Action Value')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('delta_EE_cloth_replay_rtc.png')
    print(f"plot saved to delta_EE_cloth_replay_rtc.png")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))