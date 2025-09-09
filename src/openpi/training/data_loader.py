from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar
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
from x2robot_dataset.dynamic_robot_dataset import DynamicRobotDataset
from x2robot_dataset.common.constants import ACTION_KEY_RANGES

import jax
import jax.numpy as jnp
# import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
import logging

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

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


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

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


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
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
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


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


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)

def default(config:OmegaConf, attribute_level:str, default_value):
    return OmegaConf.select(config, attribute_level, default=default_value)

def create_x2robot_dataloader(cfg):
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
    model_type = default(cfg, 'model.model_type', None)

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
    dropout_agent_pos_ratio = default(cfg, 'task.dropout_agent_pos_ratio', 0.0)
    instruction_key = default(cfg, 'task.instruction_keys', ['general'])
    select_high_quality_data = default(cfg, 'task.select_high_quality_data', False)
    buffer_size = 1000 if cfg.training.debug else 15000

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
        dropout_agent_pos_ratio=dropout_agent_pos_ratio,
        select_high_quality_data=select_high_quality_data,
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
    if custon_normalization_path is not None:
        with open(custon_normalization_path, 'r') as f:
            import json
            custom_norm_stats = json.load(f)
        norm_stats['action_mean'] = np.array(custom_norm_stats['norm_stats']['action']['mean'])
        norm_stats['action_std'] = np.array(custom_norm_stats['norm_stats']['action']['std'])
        norm_stats['state_mean'] = np.array(custom_norm_stats['norm_stats']['agent_pos']['mean'])
        norm_stats['state_std'] = np.array(custom_norm_stats['norm_stats']['agent_pos']['std'])
        print(f"Using custom normalization stats from {custon_normalization_path}")
    else:
        norm_stats['action_mean'] = np.array(predict_action_min)
        norm_stats['action_std'] = np.array(predict_action_max) - np.array(predict_action_min)
        norm_stats['state_mean'] = np.array(agent_pos_min)
        norm_stats['state_std'] = np.array(agent_pos_max) - np.array(agent_pos_min)
    data_config.update(
        norm_stats=norm_stats,
    )

    # TODO: Add extra action dims. e.g. 6D + relative action = 20

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
        buffer_size=buffer_size, # Only use 1000 for debug mode
        device='jax',
        model=model_type,
    )
    train_num = dataset.global_train_iters.value
    val_num = dataset.global_val_iters.value
    total_frames = train_num * batch_size * jax.process_count()
    total_frames_val = val_num * batch_size * jax.process_count()
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
        logging.info(f"rank {jax.process_index()}: All processes synchronized after dataset initialization")
    
    train_dataloader = dataset.get_train_dataloader()
    val_dataloader = dataset.get_val_dataloader()
    return dataset, train_dataloader, val_dataloader

class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
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

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
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
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
