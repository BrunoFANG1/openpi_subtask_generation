import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device='cuda:6'

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
    'follow_left_ee_cartesian_pos':  {'name': 'follow1_pos', 'shape': 3, 'slice': slice(0, 3)},
    'master_left_ee_rotation':       {'name': 'follow1_pos', 'shape': 3, 'slice': slice(3, 6)},
    'follow_left_ee_rotation':       {'name': 'follow1_pos', 'shape': 3, 'slice': slice(3, 6)},
    'follow_left_gripper':           {'name': 'follow1_pos', 'shape': 1, 'slice': slice(6, 7)},
    'master_right_ee_cartesian_pos': {'name': 'follow2_pos', 'shape': 3, 'slice': slice(0, 3)},
    'follow_right_ee_cartesian_pos': {'name': 'follow2_pos', 'shape': 3, 'slice': slice(0, 3)},
    'master_right_ee_rotation':      {'name': 'follow2_pos', 'shape': 3, 'slice': slice(3, 6)},
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

class PromptManager:
    """Manages prompt switching via keyboard input."""
    
    def __init__(self, prompts):
        self.prompts = prompts
        self.current_prompt_index = 0
        self.current_prompt = prompts[0] if prompts else ""
        self.running = True
        self.lock = threading.Lock()
        
        # Start keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        print("Prompt Manager initialized!")
        print("Available prompts:")
        for i, prompt in enumerate(self.prompts):
            print(f"  {i+1}: {prompt}")
        print("Press number keys (1-9) to switch prompts, 'q' to quit")
    
    def _keyboard_listener(self):
        """Listen for keyboard input in a separate thread."""
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            
            while self.running:
                # Check if there's input available
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    
                    if char == 'q':
                        print("\nQuitting...")
                        self.running = False
                        break
                    elif char.isdigit():
                        key_num = int(char)
                        if 1 <= key_num <= len(self.prompts):
                            with self.lock:
                                self.current_prompt_index = key_num - 1
                                self.current_prompt = self.prompts[self.current_prompt_index]
                            print(f"\nSwitched to prompt {key_num}: {self.current_prompt}")
                        else:
                            print(f"\nInvalid prompt number: {key_num}. Available: 1-{len(self.prompts)}")
                            
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def get_current_prompt(self):
        """Get the current prompt in a thread-safe way."""
        with self.lock:
            return self.current_prompt
    
    def stop(self):
        """Stop the keyboard listener."""
        self.running = False

def normalize_action(action_input, min_range, max_range):
    return (action_input - min_range) / (max_range - min_range)

def unnormalize_action(action_input, min_range, max_range):
    return action_input * (max_range - min_range) + min_range

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
    openloop_filepath: str | None = '/home/fangxinyuan/projects/dataset/test_dataset'

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
    setting = "test"
    os.makedirs(f"saved/{setting}", exist_ok=True)
    args.action_interpolate_multiplier = 10
    args.action_start_ratio = 0.0
    args.action_end_ratio = 1.0
    save_path = f"saved/{setting}/"

    # Define your prompt list here
    # prompt_list = [
    #     "抓住香包开口的一侧，拿起香包",
    #     "从木盒里拿起内胆",
    #     "把内胆放进香包里",
    #     "抓住香包开口两侧的绳头，拉紧",
    #     "把香包正面朝上的放到桌上",
    #     "拿起徽章用力按到香包正面的魔术贴上",
    #     "拿起香包，放到对面人伸出的手上"
    # ]
    prompt_list = [
        "制作香包，要黑色内胆",
        "制作香包，要粉色内胆",
        "制作香包，要绿色内胆",
        "制作香包，要黄色内胆",
    ]
    prompt_list = [ 
        "拿走粉红色篮子，右转90度，放到小货架上，换新的粉红色篮子，回到起点",
        "拿走绿色篮子，右转90度，放到小货架上，换新的绿色篮子，回到起点"
        "从桌面拿走木头盒子，右转90度，放到小货架上，换新的木头盒子，回到起点",
        "从大货架上拿走绿色香包篮子，左转90度，放到白色桌子上",
        "从大货架上拿走白色内胆篮子，左转90度，放到白色桌子上",
        "从大货架上拿走绿色徽章篮子，左转90度，放到白色桌子上",
    ]
    # Initialize prompt manager
    prompt_manager = PromptManager(prompt_list)

    policy = create_policy(args)

    obs_action_keys = [
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation",
        "follow_left_gripper",
        "follow_right_ee_cartesian_pos",
        "follow_right_ee_rotation",
        "follow_right_gripper",
        "follow_left_gripper_cur",
        "follow_right_gripper_cur",
        "velocity_decomposed",
        "height",
        "head_rotation"
    ]
    predict_action_keys = [
        "master_left_ee_cartesian_pos",
        "master_left_ee_rotation",
        "follow_left_gripper",
        "master_right_ee_cartesian_pos",
        "master_right_ee_rotation",
        "follow_right_gripper",
        "velocity_decomposed",
        "height",
        "head_rotation",
    ]
    predict_action_masks = build_action_masks(predict_action_keys, MODEL_TO_ROBOT_MAPPING)
    predict_action_min, predict_action_max, agent_pos_min, agent_pos_max = [], [], [], []
    for key in predict_action_keys:
        predict_action_min += ACTION_KEY_RANGES[key]['min_range']
        predict_action_max += ACTION_KEY_RANGES[key]['max_range']
    for key in obs_action_keys:
        agent_pos_min += ACTION_KEY_RANGES[key]['min_range']
        agent_pos_max += ACTION_KEY_RANGES[key]['max_range']
    predict_action_min = np.array(predict_action_min)
    predict_action_max = np.array(predict_action_max)
    agent_pos_min = np.array(agent_pos_min)
    agent_pos_max = np.array(agent_pos_max)
    obs_action_dim = sum(MODEL_TO_ROBOT_MAPPING[key]['shape'] for key in obs_action_keys)

    def _pred_func(self, views, actions) -> dict:
        if not hasattr(self, 'cur_velocity_decomposed'):
            self.cur_velocity_decomposed = np.array([0.0, 0.0, 0.0])
        obs_list = []
        for key in obs_action_keys:
            mapping_info = MODEL_TO_ROBOT_MAPPING[key]
            robot_key, data_slice = mapping_info['name'], mapping_info.get('slice')
            if robot_key not in actions:
                raise ValueError(f"Required robot key '{robot_key}' for model key '{key}' not in received actions")
            if key == 'velocity_decomposed':
                obs_list.append(self.cur_velocity_decomposed)
            else:
                raw_data = actions[robot_key]
                if isinstance(raw_data, float) or isinstance(raw_data, int):
                    raw_data = np.array([raw_data])
                obs_list.append(np.array(raw_data[data_slice] if data_slice else raw_data))
            # print(f'{key}: {obs_list[-1].shape}')
        
        agent_pos = np.concatenate(obs_list)
        if agent_pos.shape[-1] != obs_action_dim:
            raise ValueError(f'agent_pos dim mismatch! Got {agent_pos.shape[-1]}, expected {obs_action_dim}')
        
        obs_list = []
        for key in predict_action_keys:
            mapping_info = MODEL_TO_ROBOT_MAPPING[key]

        normlized_agent_data = normalize_action(agent_pos, agent_pos_min, agent_pos_max)
        current_prompt = prompt_manager.get_current_prompt()  # Use first prompt for now
        # current_prompt = prompt_list[0]  # Use first prompt for now
        print(f'{current_prompt}')
        obs = {
            'images': {
                'left_wrist_view': np.array(views['camera_left'])[0],
                'face_view': np.array(views['camera_front'])[0],
                'right_wrist_view': np.array(views['camera_right'])[0],
            },
            'prompt': current_prompt,
            'state': normlized_agent_data,
        }

        start_time = time.time()
        result = policy.infer(obs)
        end_time = time.time()
        print(f'inference time: {end_time - start_time}')
        action_pred = result['actions']
        action_pred = unnormalize_action(action_pred, predict_action_min, predict_action_max)


        # Parse predicted action tensor
        parsed_actions = {key: action_pred[:, predict_action_masks[key]] for key in predict_action_keys}
        num_steps = action_pred.shape[0]

        # Process predicted actions
        current_car_pose = np.array(actions.get('car_pose', [0,0,0]))
        target_car_poses = np.tile(current_car_pose, (num_steps, 1))
        
        if 'velocity_decomposed' in parsed_actions:
            velocities = parsed_actions['velocity_decomposed']
            temp_poses = []
            for i in range(num_steps):
                current_car_pose = velocity_to_pose(velocities[i,0], velocities[i,1], velocities[i,2], 0.05, current_car_pose)
                temp_poses.append(current_car_pose)
            target_car_poses = np.array(temp_poses)
            self.cur_velocity_decomposed = velocities[-1]



        def assemble_arm_action(side):
            """
            Assembles the full arm action by dynamically checking for predicted 
            master or follow keys.
            """
            # Dynamically determine which position and rotation keys to use based on what the model predicted.
            pos_key = f'master_{side}_ee_cartesian_pos'
            if pos_key not in parsed_actions:
                pos_key = f'follow_{side}_ee_cartesian_pos'

            rot_key = f'master_{side}_ee_rotation'
            if rot_key not in parsed_actions:
                rot_key = f'follow_{side}_ee_rotation'
            
            # Gripper key is consistently 'follow' in most configs.
            grip_key = f'follow_{side}_gripper'
            
            # Collect the predicted parts that are available in parsed_actions.
            # The order of keys here is important to form the 7D action vector correctly.
            parts = []
            for key in [pos_key, rot_key, grip_key]:
                if key in parsed_actions:
                    parts.append(parsed_actions[key])
            
            return np.concatenate(parts, axis=1) if parts else None
        follow1 = assemble_arm_action('left')
        follow2 = assemble_arm_action('right')
        lift = parsed_actions.get('height')
        head_rotation = parsed_actions.get('head_rotation')

        # Interpolate and slice
        inter_len = args.action_interpolate_multiplier * num_steps
        start_frame, end_frame = int(args.action_start_ratio * inter_len), int(args.action_end_ratio * inter_len)
        
        def interpolate_arm(data, default_val):
            if data is None:
                return [default_val] * (end_frame - start_frame)
            # Use the specialized interpolation function for 7D arm actions
            interp = interpolates_actions(data, num_steps, inter_len, data.shape[1])
            return interp[start_frame:end_frame].tolist()
            
        def interpolate_simple(data, default_val):
            if data is None:
                return [default_val] * (end_frame - start_frame)
            # Use simple linear interpolation for other values (car, lift, head)
            original_indices = np.linspace(0, 1, num_steps)
            target_indices = np.linspace(0, 1, inter_len)
            interp_data = np.zeros((inter_len, data.shape[1]))
            for i in range(data.shape[1]):
                interp_data[:, i] = np.interp(target_indices, original_indices, data[:, i])
            return interp_data[start_frame:end_frame].tolist()
        # print(f'lift:{lift.shape}')
        # import pdb; pdb.set_trace()
        return {
            "follow1_pos": interpolate_arm(follow1, actions.get('follow1_pos')),
            "follow2_pos": interpolate_arm(follow2, actions.get('follow2_pos')),
            # "car_pose": [actions['car_pose'] for e in interpolate_simple(lift, [actions.get('lift', 0.4)])],
            #"lift": interpolate_simple([lift], [actions.get('lift', 0.4)]),
            # "lift": [e[0] for e in interpolate_simple(lift, [actions.get('lift', 0.4)])],
            "car_pose": interpolate_simple(target_car_poses, actions.get('car_pose')),
            #"lift": interpolate_simple([lift], [actions.get('lift', 0.4)]),
            "lift": [e[0] for e in interpolate_simple(lift, [actions.get('lift', 0.4)])],
            "head_pos": interpolate_simple(head_rotation, actions.get('head_pos')),
            "follow1_joints": [], "follow2_joints": [], # Not handled
        }

    RobotController.prediction = _pred_func
    
    # Add debug output for robot connection
    print("Creating robot controller with robot_id=7...")
    robot_controller = RobotController(robot_id=8, max_time_step=100000)
    
    # Debug: Check robot info before connecting
    print(f"Robot info: {robot_controller.robot_comm.robot_info}")
    if robot_controller.robot_comm.robot_info:
        print(f"Host: {robot_controller.robot_comm.robot_info['host']}")
        print(f"Action port: {robot_controller.robot_comm.robot_info['action_port']}")
        print(f"Keyboard port: {robot_controller.robot_comm.robot_info['keyboard_port']}")
        print(f"Stop port: {robot_controller.robot_comm.robot_info['stop_port']}")
    
    print("Connecting to robot...")
    robot_controller.connect()
    print("robot connected")
    robot_controller.run(record_mode = False)
    print("robot closing")
    robot_controller.close()
    
    # Clean up prompt manager
    prompt_manager.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))