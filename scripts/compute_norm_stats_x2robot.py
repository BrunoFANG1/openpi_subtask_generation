import os
os.environ["HF_LEROBOT_HOME"] = "/x2robot_v2/xinyuanfang/projects_v2/.cache/lerobot"
# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['OPENPI_DATA_HOME'] = '/x2robot_v2/xinyuanfang/projects_v2/.cache/openpi'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT", "12355")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc
import time
import hydra
import signal
import pathlib
import dataclasses
import functools
import logging
import platform
import numpy as np
from typing import Any
from pathlib import Path
from jax.experimental import multihost_utils

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
from openpi.models.model import Observation
from hydra.utils import instantiate
from x2robot_dataset.common.constants import ACTION_KEY_RANGES
from x2robot_dataset.lazy_dataset import (
    X2RDataChunkConfig,
    X2RDataProcessingConfig,
)
from x2robot_dataset.dynamic_robot_dataset import DynamicRobotDataset
from x2robot_dataset.common.constants import ACTION_KEY_RANGES
from omegaconf import OmegaConf

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

def default(config:OmegaConf, attribute_level:str, default_value):
    return OmegaConf.select(config, attribute_level, default=default_value)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):

    if int(os.environ.get("SLURM_NTASKS", "0")) > 1:
        jax.distributed.initialize()
    # Set master addr and port after jax distributed initialization
    if MASTER_ADDR:
        os.environ['MASTER_ADDR'] = MASTER_ADDR
    if MASTER_PORT:
        os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    logging.info(f"Running on: {platform.node()}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("/x2robot_v2/xinyuanfang/projects_v2/.cache/openpi/jax").expanduser()))

    # Create dataloader
    train_test_split = default(cfg, "task.dataset.train_val_split", 0.9)

    # configure dataset
    horizon = default(cfg, "task.action_horizon", 20)
    action_history_length = default(cfg, "task.action_history_length", 0)
    image_history_length = default(cfg, "task.image_history_length", 0)
    trim_stationary = default(cfg, 'task.trim_stationary', False) # 是否去除静止动作
    filter_angle_outliers = default(cfg, "task.filter_angle_outliers", True)  # 是否过滤角度异常值, 默认要过滤
    sample_rate = default(cfg, "task.dataset.sample_rate", 1.0)  # 针对action和image的采样率
    cache_dir = default(cfg, "task.dataset.cache_dir", "/x2robot_v2/Data/.cache/dataset_cache")  # 数据集根目录
    dataset_config_path = default(cfg, "task.task_config_path", None)  # 数据集配置文件路径
    assert dataset_config_path is not None, f"dataset_config_path is None, please check your config file"
    
    default_instruction = default(cfg, 'task.dataset.instruction', '')
    instruction_path = default(cfg, 'task.dataset.instruction_path', None)
    instruction_key = default(cfg, 'task.dataset.instruction_key', None)
    one_by_one_relative = default(cfg, 'task.dataset.one_by_one_relative', False)
    
    print(f"instruction_key配置: {instruction_key}")
    print(f"instruction_path配置: {instruction_path}")
    
    batch_size = cfg.train_dataloader.batch_size

    # 从shape_meta中构建cam_mapping - 配置化方式
    # camera_name -> obs_key
    cam_mapping = {}
    obs_shape_meta = cfg.task.shape_meta["obs"]
    
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            camera_name = attr.get("camera_name", None)
            if camera_name is not None:
                cam_mapping[camera_name] = key
                print(f"Added cam mapping: {camera_name} -> {key}")
            else:
                print(f"Warning: RGB observation {key} missing camera_name")

    
    print(f"Final cam_mapping: {cam_mapping}")
    merge_cur_history = action_history_length > 0  # agent_pos里是否加入动作历史
    merge_image_history = image_history_length > 0  # 观测图像里是否加入图像历史

    # 直接从任务配置中获取action keys
    predict_action_keys = cfg.task.predict_action_keys
    obs_action_keys = cfg.task.obs_action_keys
    
    # 验证配置
    assert predict_action_keys is not None, "predict_action_keys must be configured in task config"
    assert obs_action_keys is not None, "obs_action_keys must be configured in task config"

    use_custom_action_data_path = default(cfg, 'task.use_custom_action_data_path', False)
    global_action_data_base_path = default(cfg, 'task.global_action_data_base_path', None)
    ignore_prediction_keys = default(cfg, 'task.ignore_prediction_keys', [])
    detect_motion = default(cfg, 'task.detect_motion', True)
    custon_normalization_path = default(cfg, 'task.custon_normalization_path', None)
    distributed_instruction_ratio = default(cfg, 'task.distributed_instruction_ratio', 1.0)

    # configure dataset
    data_config = X2RDataProcessingConfig()
    data_config.update(
        cam_mapping=cam_mapping,
        class_type="x2",
        train_test_split=train_test_split,
        filter_angle_outliers=filter_angle_outliers,
        sample_rate=sample_rate,
        parse_tactile=False,
        predict_action_keys=predict_action_keys,  # 直接使用配置
        obs_action_keys=obs_action_keys,          # 直接使用配置
        trim_stationary=trim_stationary,
        cache_dir=cache_dir,
        default_instruction=default_instruction,
        instruction_path=instruction_path,
        instruction_key=instruction_key,
        one_by_one_relative=one_by_one_relative,
        use_custom_action_data_path=use_custom_action_data_path,
        global_action_data_base_path=global_action_data_base_path,
        ignore_prediction_keys=ignore_prediction_keys,
        distributed_instruction_ratio=distributed_instruction_ratio,
        custon_normalization_path=custon_normalization_path,
    )

    data_chunk_config = X2RDataChunkConfig().update(
        left_padding=True if action_history_length > 0 else False,
        right_padding=True,
        predict_action_keys=predict_action_keys,
        action_horizon=horizon,
        obs_action_keys=obs_action_keys,
        action_history_length=action_history_length,
        image_history_length=image_history_length,
        merge_cur_history=merge_cur_history,
        merge_image_history=merge_image_history,
    )
    
    dataset = DynamicRobotDataset(
        dataset_config_path=dataset_config_path,
        data_config=data_config,
        data_chunk_config=data_chunk_config,
        rank=jax.process_index(),
        world_size=jax.process_count(),
        batch_size=batch_size,
        buffer_size=300,
        device=None,
    )
    train_num = dataset.global_train_iters.value
    val_num = dataset.global_val_iters.value
    total_frames = train_num * batch_size * jax.process_count()
    total_frames_val = val_num * batch_size * jax.process_count()
    max_frames = total_frames // 15
    # 计算train/val step
    global_batch_size = batch_size * jax.process_count()
    print(
        f"rank {jax.process_index()} total_frames:{total_frames} total_frames_val:{total_frames_val} train_num {train_num}, val_num {val_num}",
        flush=True,
    )
    print(f"rank {jax.process_index} batch_size_per_rank {batch_size} global_batch_size {global_batch_size}", flush=True)
    
    # set wall for jax distributed process
    if jax.process_count() > 1:
        # Synchronize all processes to ensure dataset is properly initialized across all ranks
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("Dataset initialization complete")
        print(f"rank {jax.process_index()}: All processes synchronized after dataset initialization", flush=True)
    
    data_loader = dataset.get_train_dataloader()

    keys = ["action", "agent_pos",]
    # import pdb; pdb.set_trace()
    stats = {key: normalize.RunningStats() for key in keys}

    count = 0
    for batch in tqdm.tqdm(data_loader, total=max_frames, desc="Computing stats"):
        for key in keys:
            if key == 'agent_pos':
                batch = batch['obs']
            values = np.asarray(batch[key][0])
            stats[key].update(values.reshape(-1, values.shape[-1]))
        count += 1
        if count > max_frames:
            break

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    datatime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_path = f'/x2robot_v2/xinyuanfang/norm_stats/norm_stats_{datatime}'
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    main()
