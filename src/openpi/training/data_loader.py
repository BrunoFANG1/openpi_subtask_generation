from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar
from pathlib import Path
from omegaconf import OmegaConf
from x2robot_dataset.lazy_dataset import (
    IterChunkDataset,
    X2RDataChunkConfig,
    X2RDataProcessingConfig,
)
from x2robot_dataset.common.data_preprocessing import _CAM_MAPPING
from x2robot_dataset.dataloader import DynamicDataLoader
from x2robot_dataset.common.collate_fn import collate_wrapper

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, local_files_only=True)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)

def default(config:OmegaConf, attribute_level:str, default_value):
    return OmegaConf.select(config, attribute_level, default=default_value)

def create_x2robot_dataloader(
    cfg: dict,
    jax_process_id: int,
    collate_type: str = 'chunking',
) -> DataLoader[dict]:

    # Ruyi 逆天代码
    rgb_keys = list()
    lowdim_keys = list()
    tactile_keys = list()
    obs_shape_meta = cfg.task.shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
        elif type == 'tactile':
            tactile_keys.append(key)
    use_tactile = len(tactile_keys) > 0
    action_dim = cfg.task.shape_meta['action'].shape[0]
    low_dim_obs_horizon = cfg.task.low_dim_obs_horizon
    img_obs_horizon = cfg.task.img_obs_horizon
    action_dim = cfg.task.shape_meta['action'].shape[0]
    parse_head_action = default(cfg, 'task.parse_head_action', False) 
    parse_head_action_v2 = default(cfg, 'task.parse_head_action_v2', 0) # 0： 不解析，1：x+0.15,z+0.2, 同时解析head_yaw,head_pitch, 2：只x+0.15,z+0.2 
    train_test_split = default(cfg, 'task.dataset.train_val_split', 0.9)
    print(f'parse_head_action:{parse_head_action}, parse_head_action_v2:{parse_head_action_v2}')
    use_quaternion = default(cfg, 'task.use_quaternion', False) # 是否使用四元数表示旋转
    relative_action = default(cfg, 'task.relative_action', False) # 是否使用相对动作
    print(f'use_quaternion:{use_quaternion}, relative_action:{relative_action}')

    # configure dataset
    low_dim_obs_horizon = default(cfg, 'task.low_dim_obs_horizon', 1)
    img_obs_horizon = default(cfg, 'task.img_obs_horizon', 1)
    horizon = default(cfg, 'task.action_horizon', 20)
    is_bi_mode = default(cfg, 'task.dataset.is_bi_mode', True)
    action_history_length = default(cfg, 'task.action_history_length', 0)
    image_history_length = default(cfg, 'task.image_history_length', 0)
    is_binocular = default(cfg, 'task.is_binocular', False) # 是否是双目模式
    # is_factory = default(cfg, 'task.is_factory', False) # 是否是工厂模式, 已经被抛弃，改成自动根据工厂手持式设备标号判定是否要加，TODO：未来需要配置化
    relative_action = default(cfg, 'task.relative_action', False) # 是否使用相对动作
    add_noise = default(cfg, 'task.add_noise', False) # 是否添加噪声, 只对相对动作有效
    filter_angle_outliers = default(cfg, 'task.filter_angle_outliers', True) # 是否过滤角度异常值, 默认要过滤
    sample_rate = default(cfg, 'task.dataset.sample_rate', 1.0) # 针对action和image的采样率
    save_meta_data = default(cfg, 'task.dataset.save_meta_data', True) # 是否保存meta数据
    force_overwrite = default(cfg, 'task.dataset.force_overwrite', False) # 是否强制覆盖
    use_gripper_cur = default(cfg, 'task.use_gripper_cur', False) # 是否使用关节力矩, 注意这里默认只使用最后一个关节的力矩
    use_joint_cur = default(cfg, 'task.use_joint_cur', False) # 是否使用观测到的所有关节的力矩
    use_diversity_file = default(cfg, 'task.use_diversity_file', False) # 是否使用多样性文件
    use_gaussian_normalization = default(cfg, 'task.use_gaussian_normalization', False) # 是否使用Z-score normalization把输入数据归一化
    use_quantile_normalization = default(cfg, 'task.use_quantile_normalization', False) # 是否使用quantile normalization把输入数据归一化
    cam_mapping = _CAM_MAPPING
    # 过滤掉不在rgb_keys里的cam
    filter_cam_mapping = {}
    for key,value in cam_mapping.items():
        if value in rgb_keys:
            filter_cam_mapping[key] = value
    cam_mapping = filter_cam_mapping
    merge_cur_history = action_history_length > 0 # agent_pos里是否加入动作历史 
    
    _ACTION_KEY_FULL_MAPPING_XY = {
        'follow_right_arm_joint_pos': 'follow_right_joint_pos',
        'follow_right_arm_joint_dev': 'follow_right_joint_dev',
        'follow_right_arm_joint_cur': 'follow_right_joint_cur',
        'follow_right_ee_cartesian_pos': 'follow_right_position',
        'follow_right_ee_rotation': 'follow_right_rotation',
        'follow_right_gripper': 'follow_right_gripper',
        'master_right_ee_cartesian_pos': 'master_right_position',
        'master_right_ee_rotation': 'master_right_rotation',
        'master_right_gripper': 'master_right_gripper',
        'follow_left_arm_joint_pos': 'follow_left_joint_pos',
        'follow_left_arm_joint_dev': 'follow_left_joint_dev',
        'follow_left_arm_joint_cur': 'follow_left_joint_cur',
        'follow_left_ee_cartesian_pos': 'follow_left_position',
        'follow_left_ee_rotation': 'follow_left_rotation',
        'follow_left_gripper': 'follow_left_gripper',
        'master_left_ee_cartesian_pos': 'master_left_position',
        'master_left_ee_rotation': 'master_left_rotation',
        'master_left_gripper': 'master_left_gripper',
    }
    full_action_keys_needed = list(_ACTION_KEY_FULL_MAPPING_XY.keys()) # Special use for Xinyuan, Note: If you change any single varaible in data_config, the dataset cache will force regenerate, which cause multi-gpu training conflicts
    prediction_action_keys = ['follow_left_ee_cartesian_pos','follow_left_ee_rotation','follow_left_gripper','follow_right_ee_cartesian_pos','follow_right_ee_rotation','follow_right_gripper']
    minmax_range_robot = default(cfg, 'task.minmax_range_robot', 'arx') # 是否是arx,leju,leju_v2
    obs_action_keys = None # obsered的action和预测的action可能不一样：设置为None则默认一样
    if minmax_range_robot == 'arx_joint':
        prediction_action_keys = ['follow_left_arm_joint_pos', 'follow_right_arm_joint_pos'] # Check normalizer for detailed explanation
    elif minmax_range_robot == 'arx_master':
        prediction_action_keys = ['master_left_ee_cartesian_pos','master_left_ee_rotation','master_left_gripper','master_right_ee_cartesian_pos','master_right_ee_rotation','master_right_gripper'] # Check normalizer for detailed explanation
    elif minmax_range_robot == 'arx_master_obs_follow':
        prediction_action_keys = ['master_left_ee_cartesian_pos','master_left_ee_rotation','master_left_gripper','master_right_ee_cartesian_pos','master_right_ee_rotation','master_right_gripper'] # Check normalizer for detailed explanation
        obs_action_keys = ['follow_left_ee_cartesian_pos','follow_left_ee_rotation','follow_left_gripper','follow_right_ee_cartesian_pos','follow_right_ee_rotation','follow_right_gripper']
    data_configs = []
    data_folders = []
    print(f'Deep Learning model needs to predict following action_keys:{prediction_action_keys}')
    for dataset_dict in cfg.task.dataset.dataset_paths:
        data_folders.append(dataset_dict['path'])
        default_instruction = dataset_dict.get('instruction', '')
        data_config = X2RDataProcessingConfig()
        data_config.update(
            cam_mapping=cam_mapping,
            default_instruction=default_instruction,
            class_type='x2',
            train_test_split=train_test_split,
            filter_angle_outliers=filter_angle_outliers,
            sample_rate=sample_rate,
            parse_tactile=use_tactile,
            action_keys=full_action_keys_needed,
        )
        data_configs.append(data_config.as_dict())
    
    data_chunk_config = X2RDataChunkConfig().update(
        left_padding=True if action_history_length > 0 else False,
        right_padding=True,
        action_horizon=horizon+1,
        action_history_length=action_history_length,
    )
    batch_size=cfg.train_dataloader.batch_size
    train_dataset = IterChunkDataset(
        data_folders,
        data_configs,
        data_chunk_config,
        preload_pool_size = 1,
        num_preloader_threads  = 1,
        max_frame_buffer_size = 2000,
        num_frame_producer_threads = 1,
        force_overwrite=force_overwrite,
        split='train',
        accelerator=None,
        rank=jax_process_id,
        world_size=jax.process_count(),
        slice_size=batch_size, # 每个process的batch_size
        root_dir=Path(cfg.data.root_dir),
        save_meta_data=save_meta_data,
        action_keys=prediction_action_keys+obs_action_keys,
        use_diversity_file=use_diversity_file,
        use_jax = True,
        action_truncated_instruction=cfg.data.action_truncated_instruction
    )
    total_frames = train_dataset.num_frames
    val_dataset = IterChunkDataset(
        data_folders,
        data_configs,
        data_chunk_config,
        preload_pool_size = 1,
        num_preloader_threads  = 1,
        max_frame_buffer_size = 2000,
        num_frame_producer_threads = 1,
        force_overwrite=force_overwrite,
        split='test',
        accelerator=None,
        rank=jax_process_id,
        world_size=jax.process_count(),
        slice_size=batch_size,
        root_dir=Path(cfg.data.root_dir),
        save_meta_data=save_meta_data,
        action_keys=prediction_action_keys+obs_action_keys,
        use_diversity_file=use_diversity_file,
        use_jax = True,
        action_truncated_instruction=cfg.data.action_truncated_instruction
    )
    total_frames_val = val_dataset.num_frames
    # 设置collate_fn
    from openpi.models import tokenizer as _tokenizer
    fast_tokenizer = _tokenizer.FASTTokenizer(250) # Don't change this
    tokenizer = _tokenizer.PaligemmaTokenizer(48) # Don't change this
    collate_fn = collate_wrapper(
        collate_type = collate_type,
        low_dim_obs_horizon=low_dim_obs_horizon,
        img_obs_horizon=img_obs_horizon,
        horizon=horizon,
        action_dim=action_dim,
        is_bi_mode=True,
        sample2instruct=None,
        to_lie_algebra=False,
        sample2imginstruct=None,
        parse_head_action=False,
        mask_type=None,
        mask_keys=None,
        merge_cur_history=merge_cur_history,
        relative_action=relative_action,
        add_noise=add_noise,
        action_keys=prediction_action_keys,
        obs_action_keys=obs_action_keys,
        use_gripper_cur=use_gripper_cur,
        use_joint_cur=use_joint_cur,
        diversity_process_fc=None if not use_diversity_file else True,
        use_jax=True,
        tokenizer=tokenizer,
        fast_tokenizer=fast_tokenizer
    )
    world_size = jax.process_count()

    # 计算train/val step
    global_batch_size = batch_size * world_size
    train_num = int(total_frames // batch_size // world_size)
    val_num = int(total_frames_val // batch_size // world_size)

    # 加载dataloader
    train_dataloader = DynamicDataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            num_workers=1,
                                            gpu_id=jax_process_id,
                                            collate_fn=collate_fn,
                                            length=train_num)
    val_dataloader = DynamicDataLoader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        num_workers=1,
                                        gpu_id=jax_process_id,
                                        collate_fn=collate_fn,
                                        length=val_num)
    return train_dataloader, val_dataloader

class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                # jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.Mesh([jax.devices()[0]], ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
