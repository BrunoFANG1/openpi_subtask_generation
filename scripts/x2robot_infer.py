import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import dataclasses
import enum
import logging
import socket
import time
from pathlib import Path

import struct
import tyro
import json
import cv2
import numpy as np
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

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
    if action_dim == 2: # 头部动作直接线性插值
        for i in range(action_dim):
            interpolated_actions[:, i] = np.interp(target_indices, original_indices, actions[:, i])
        return interpolated_actions

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

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    X2ROBOT = "x2robot"

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


import threading

class KeyboardThread(threading.Thread):

    def __init__(self, input_cbk = None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()

    def run(self):
        while True:
            self.input_cbk(input())



@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


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
    openloop_filepath: str | None = '/x2robot/zhengwei/10044/20250410-day-spell-word-2'


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
}


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

def recv_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def read_img(conn,i,save_path,count=0):
    image_size = struct.unpack('<L', conn.recv(4))[0]
    image = recvall(conn, image_size)
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("image-{i}-{count}.jpg", image) #
    return image

def normalize_action(action_input, min_range, max_range):
    return (action_input - min_range) / (max_range - min_range)

def unnormalize_action(action_input, min_range, max_range):
    return action_input * (max_range - min_range) + min_range

instruction_list = [
    'grab f and put on the board',
    'grab o and put on the board',
    'grab u and put on the board',
    'grab r and put on the board',
    ]
instruction_idx = 0

def keyboard_callback(inp):
    global instruction_idx
    if inp == '1':
        instruction_idx = (instruction_idx + 1) % len(instruction_list)
        print(f"Switched to instruction {instruction_idx}: {instruction_list[instruction_idx]}")
    elif inp == '0':
        instruction_idx = (instruction_idx - 1) % len(instruction_list)
        print(f"Switched to instruction {instruction_idx}: {instruction_list[instruction_idx]}")
    else:
        print(f"Received input: {inp}. No action taken.")

def main(args: Args) -> None:
    
    policy = create_policy(args)

    if args.openloop or args.log_replay:
        assert args.openloop_filepath is not None, "openloop_filepath must be provided when openloop is True"
        from x2robot_dataset.common.datasets import (
            create_instance,
            MultiVideoLazyDataset
        )
        from x2robot_dataset.lazy_dataset import (
            X2RDataProcessingConfig,
        )
        root_dir = Path('/x2robot/brae/Data/hf_datasets_tmp2')
        
        data_config = X2RDataProcessingConfig().as_dict()

        out_repo = create_instance(data_config).from_raw_to_videolazy_format(
            dataset_path=args.openloop_filepath,
            force_overwrite=False,
            save_meta_data=True,
            num_threads=1,
            root_dir=root_dir,
            class_type=data_config['class_type'],
        )
        out_repo['config'] = data_config
        dataset = MultiVideoLazyDataset(
            repos=[out_repo], root=root_dir, split=['train'],
        )
        
        traj_idx = 0
        traj_data = dataset[traj_idx]
        traj_frame_num = len(traj_data['observations.left_wrist_view'])

        instruction_key = 'diversity.distribute'
        instruction_seg = traj_data.get(instruction_key, )
        if isinstance(instruction_seg, str):
            instruction_seg = json.loads(instruction_seg)

        start_idx = 0
        end_idx = traj_frame_num - 1

        seg_idx = 1 # select the segment you want
        if instruction_seg:
            for idx, (split, instruction) in enumerate(instruction_seg.items()):
                if idx == seg_idx:
                    start_idx = int(split.split(' ')[0])
                    end_idx = int(split.split(' ')[1])
                    item = instruction.split('将')[1].split('放')[0]
                    item = f'grab {item} and put on the board'
                    print(f"Selected segment {seg_idx}: start_idx={start_idx}, end_idx={end_idx}, instruction={item}")
                    instruction_list.append(item)
                    break
        count = start_idx
        # traj_frame_num = dataset.frame_count_list[0][traj_idx]
    else:
        count = 0

    kthread = KeyboardThread(keyboard_callback)

    if not args.openloop:
    # if False:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(True) #设置通信是阻塞式
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ip = '10.60.251.101' #
        port = 10812
        sock.bind((ip, port))
        sock.listen(1)
        print(f"Server is listening on {ip}:{port}")

        conn, addr = sock.accept()
        print(f"Connection from {addr}")
    else:
        # plot openloop data
        ground_truth = []
        model_output = []

    max_time_step = 1000

    min_range = np.array([-0.1, -0.5, -0.5, -3.0, -3.0, -3.0 , -9, -0.1, -0.5, -0.5, -3.0, -3.0, -3.0 , -9], dtype=np.float32)
    max_range = np.array([0.5,  0.5,  0.5, 3.0, 3.0, 3.0, 9,0.5,  0.5,  0.5, 3.0, 3.0, 3.0, 9], dtype=np.float32)

    while True:
        if count>max_time_step:
            break

        print(instruction_list[instruction_idx])

        if args.openloop or args.log_replay:
            if count >= end_idx:
                print(f"End of trajectory data, count: {count}")
                break

        if not args.openloop:
            data_size = struct.unpack('<L', conn.recv(4))[0]
            # data = conn.recv(data_size)
            data = recv_all(conn, data_size)
            action_data = json.loads(data.decode('utf8'))

            left_agent_data = action_data['follow1_pos'] # (7)
            right_agent_data = action_data['follow2_pos'] # (7)

            save_path = 'None'
            image1 = read_img(conn,1,save_path,count) # left
            image2 = read_img(conn,2,save_path,count) # front
            image3 = read_img(conn,3,save_path,count) # right
        
        else:
            image1 = traj_data['observations.left_wrist_view'][count] # left
            image2 = traj_data['observations.face_view'][count] # front
            image3 = traj_data['observations.right_wrist_view'][count] # right
            left_ee_pos = traj_data['actions.follow_left_ee_cartesian_pos'][count]
            left_ee_rot = traj_data['actions.follow_left_ee_rotation'][count]
            left_gripper = traj_data['actions.follow_left_gripper'][count]
            left_agent_data = np.concatenate([left_ee_pos, left_ee_rot, left_gripper], axis=0)
            right_ee = traj_data['actions.follow_right_ee_cartesian_pos'][count]
            right_ee_rot = traj_data['actions.follow_right_ee_rotation'][count]
            right_gripper = traj_data['actions.follow_right_gripper'][count]
            right_agent_data = np.concatenate([right_ee, right_ee_rot, right_gripper], axis=0)

        # if fake_data:
        #     image1 = np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
        #     image2 = np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
        #     image3 = np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
        #     left_agent_data = np.random.rand(7)
        #     right_agent_data = np.random.rand(7)

        

        if args.log_replay:
            move_steps = 10
            follow_left_ee = traj_data['actions.follow_left_ee_cartesian_pos'][count:count+move_steps]
            follow_right_ee = traj_data['actions.follow_right_ee_cartesian_pos'][count:count+move_steps]
            follow_left_rot = traj_data['actions.follow_left_ee_rotation'][count:count+move_steps]
            follow_right_rot = traj_data['actions.follow_right_ee_rotation'][count:count+move_steps]
            follow_left_gripper = traj_data['actions.follow_left_gripper'][count:count+move_steps]
            follow_right_gripper = traj_data['actions.follow_right_gripper'][count:count+move_steps]
            action_pred = np.concatenate([follow_left_ee, follow_left_rot, follow_left_gripper, follow_right_ee, follow_right_rot, follow_right_gripper], axis=-1)
        else:

            h,w,c = np.array(image1).shape
            camera_front = np.array(image2).reshape(h,w,c)
            camera_left = np.array(image1).reshape(h,w,c)
            camera_right = np.array(image3).reshape(h,w,c)

            state = np.concatenate([left_agent_data, right_agent_data])
            norm_state = normalize_action(state, min_range, max_range)
            obs = {
                'images': {
                    'left_wrist_view': camera_left,
                    'face_view': camera_front,
                    'right_wrist_view': camera_right,
                },
                'prompt': instruction_list[instruction_idx],
                'state': norm_state,
            }
            time_preprocess = time.time()
            # config transforms include data.transform (state padding, resizing img), normalize and denormalize will not be applied; 
            # normalization have been done in dataloader (collate_fn) when training 
            action_pred = policy.infer(obs)
            action_pred = action_pred['actions']
            action_pred = unnormalize_action(action_pred, min_range, max_range)
            
            move_steps = action_pred.shape[0] // 2
            action_pred = action_pred[:move_steps, ...]

        if args.openloop:
            master_left_ee = traj_data['actions.master_left_ee_cartesian_pos'][count:count+move_steps]
            master_right_ee = traj_data['actions.master_right_ee_cartesian_pos'][count:count+move_steps]
            master_left_rot = traj_data['actions.master_left_ee_rotation'][count:count+move_steps]
            master_right_rot = traj_data['actions.master_right_ee_rotation'][count:count+move_steps]
            master_left_gripper = traj_data['actions.master_left_gripper'][count:count+move_steps]
            master_right_gripper = traj_data['actions.master_right_gripper'][count:count+move_steps]
            ground_truth.append(np.concatenate([master_left_ee, master_left_rot, master_left_gripper, master_right_ee, master_right_rot, master_right_gripper], axis=-1))
            model_output.append(action_pred)

        else:
            # interpolates actions
            actions_factor = 10
            action_num = action_pred.shape[0]
            left_action_pred = interpolates_actions(actions=action_pred[:,:7], num_actions=action_pred.shape[0], target_num_actions=actions_factor*action_num, action_dim=7)
            right_action_pred = interpolates_actions(actions=action_pred[:,7:14], num_actions=action_pred.shape[0], target_num_actions=actions_factor*action_num, action_dim=7)
            # time_infer = time.time()
            action_pred = np.concatenate([left_action_pred,right_action_pred], axis=1)
            # print(f'infer time: {time_infer-time_preprocess}')

            follow1 = action_pred[:, :7] # left EEF
            follow2 = action_pred[:, 7:14] # right EEF



            follow1_pos = follow1.tolist()
            follow2_pos = follow2.tolist()
            data_dir ={"follow1_pos":follow1_pos,"follow2_pos":follow2_pos}
            data_str = json.dumps(data_dir)
            data_bytes = data_str.encode('utf-8')

            # time_postprocess = time.time()
            # print(f'post process time: {time_postprocess-time_infer}')
            
            conn.sendall(struct.pack('<L', len(data_bytes)))
            conn.sendall(data_bytes)
        
        count += move_steps

    if args.openloop:
        ground_truth = np.stack(ground_truth, axis=0)
        model_output = np.stack(model_output, axis=0)

        dim = ground_truth.shape[-1]

        ground_truth = ground_truth.reshape(-1, dim)
        model_output = model_output.reshape(-1, dim)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4*dim))

        for i in range(dim):
            plt.subplot(dim, 1, i + 1)

            # plot every 10th action
            plt.xticks(np.arange(0, len(ground_truth), step=10))

            plt.plot(ground_truth[:, i], label='Ground Truth', color='blue')
            plt.plot(model_output[:, i], label='Model Output', color='orange')
            plt.title(f'Action Dimension {i + 1}')
            plt.xlabel('Time Step')
            plt.ylabel('Action Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('openloop_action_comparison.png')
        
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
