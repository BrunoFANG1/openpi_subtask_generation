import dataclasses
import enum
import logging
import socket
import time

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


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


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


def main(args: Args) -> None:
    
    policy = create_policy(args)
    policy_metadata = policy.metadata

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(True) #设置通信是阻塞式
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ip = '10.60.251.101'
    port = 10812
    sock.bind((ip, port))
    sock.listen(1)
    print(f"Server is listening on {ip}:{port}")

    conn, addr = sock.accept()
    print(f"Connection from {addr}")
    

    # # Record the policy's behavior.
    # if args.record:
    #     policy = _policy.PolicyRecorder(policy, "policy_records")
    count = 0
    max_time_step = 1000
    while True:
        if count>max_time_step:
            break
        count += 1
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

        ##################################################################
        # image1 = np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
        # image2 = np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
        # image3 = np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
        # left_agent_data = np.random.rand(7)
        # right_agent_data = np.random.rand(7)
        ##################################################################


        h,w,c = np.array(image1).shape
        camera_front = np.array(image2).reshape(h,w,c)
        camera_left = np.array(image1).reshape(h,w,c)
        camera_right = np.array(image3).reshape(h,w,c)

        state = np.concatenate([left_agent_data, right_agent_data])

        obs = {
            'images': {
                'left_wrist_view': camera_left,
                'face_view': camera_front,
                'right_wrist_view': camera_right,
            },
            'prompt': '',
            'state': state,
        }
        time_preprocess = time.time()

        action_pred = policy.infer(obs)
        action_pred = action_pred['actions'][0:30]
        print(action_pred)
        # assert False

        actions_factor = 10
        # interpolates actions
        action_num = action_pred.shape[0]
        left_action_pred = interpolates_actions(actions=action_pred[:,:7], num_actions=action_pred.shape[0], target_num_actions=actions_factor*action_num, action_dim=7)
        right_action_pred = interpolates_actions(actions=action_pred[:,7:14], num_actions=action_pred.shape[0], target_num_actions=actions_factor*action_num, action_dim=7)
        time_infer = time.time()
        action_pred = np.concatenate([left_action_pred,right_action_pred], axis=1)
        print(f'infer time: {time_infer-time_preprocess}')
        # assert False

        follow1 = action_pred[:, :7] # left EEF
        follow2 = action_pred[:, 7:14] # right EEF



        follow1_pos = follow1.tolist()
        follow2_pos = follow2.tolist()

        data_dir ={"follow1_pos":follow1_pos,"follow2_pos":follow2_pos}
        data_str = json.dumps(data_dir)
        data_bytes = data_str.encode('utf-8')
        time_postprocess = time.time()
        print(f'post process time: {time_postprocess-time_infer}')
        conn.sendall(struct.pack('<L', len(data_bytes)))
        conn.sendall(data_bytes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
