import os
os.environ["HF_LEROBOT_HOME"] = "/x2robot_v2/xinyuanfang/projects_v2/.cache/lerobot"
# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['OPENPI_DATA_HOME'] = '/x2robot_v2/xinyuanfang/projects_v2/.cache/openpi'

MASTER_ADDR = os.environ.get("MASTER_ADDR", None)
MASTER_PORT = os.environ.get("MASTER_PORT", None)

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

def sigterm_handler(signum, frame):
    logging.info(f"Process {jax.process_index()} received SIGTERM, exiting")
    os._exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)


def init_logging(debug=False):
    """Custom logging format for better readability. Only logs from main process."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    # Only set up logging for the main process
    if jax.process_index() == 0:
        formatter = CustomFormatter(
            fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s (%(process)d:%(filename)s:%(lineno)s)",
            datefmt="%H:%M:%S",
        )

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.handlers[0].setFormatter(formatter)
    else:
        # For non-main processes, set the root logger to a high level to suppress most messages
        logging.getLogger().setLevel(logging.ERROR)
        
        # Create a null handler to avoid "No handler found" warnings
        null_handler = logging.NullHandler()
        logging.getLogger().addHandler(null_handler)
        
        # Optionally, you can also remove existing handlers
        for handler in logging.getLogger().handlers[:]:
            if not isinstance(handler, logging.NullHandler):
                logging.getLogger().removeHandler(handler)

def convert_per_process_batch_to_jax(data_dict, data_sharding):
    """Convert NumPy arrays to JAX arrays and create distributed arrays."""
    # Define a function to convert arrays to distributed arrays
    def create_distributed_array(x):
        if isinstance(x, (np.ndarray, jnp.ndarray, jax.Array)):
            if jax.process_count() > 1:
                return jax.make_array_from_process_local_data(
                    sharding=data_sharding,
                    local_data=x
                )
            elif jax.process_count() == 1:
                return jax.device_put(x, data_sharding)
        elif isinstance(x, dict):
            logging.info(f"Converting dict to distributed arrays of keys: {x.keys()}")
            return {k: create_distributed_array(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [create_distributed_array(v) for v in x]
        else:
            return x
    
    # Convert all arrays to distributed arrays
    return jax.tree.map(create_distributed_array, data_dict)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            # mode='offline'
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    logging.info(f"Jit compiling init")
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding

@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info

def default(config:OmegaConf, attribute_level:str, default_value):
    return OmegaConf.select(config, attribute_level, default=default_value)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):

    # Create a complete TrainConfig with all required parameters
    config = _config.TrainConfig(
        name=cfg.config_name,
        exp_name=cfg.exp_name,
        model=instantiate(cfg.model.model_config),  # Use Hydra's instantiate to create Pi0Config instance
        weight_loader=instantiate(cfg.model.weight_loader),
        data=instantiate(cfg.model.data),
        num_train_steps=cfg.model.num_train_steps,
        lr_schedule=instantiate(cfg.lr_schedule) if hasattr(cfg, 'lr_schedule') else _optimizer.CosineDecaySchedule(),
        optimizer=instantiate(cfg.optimizer) if hasattr(cfg, 'optimizer') else _optimizer.AdamW(),
        batch_size=cfg.train_dataloader.batch_size,
        num_workers=cfg.train_dataloader.get('num_workers', 2),
        log_interval=cfg.training.get('log_interval', 100),
        save_interval=cfg.training.get('save_interval', 1000),
        keep_period=cfg.checkpoint.get('keep_period', 1000),
        overwrite=cfg.training.get('overwrite', False),
        resume=cfg.training.get('resume', False),
        wandb_enabled=cfg.logging.get('enabled', True),
        fsdp_devices=cfg.training.get('fsdp_devices', 1),
        seed=cfg.training.seed,
    )

    if int(os.environ.get("SLURM_NTASKS", "0")) > 1:
        jax.distributed.initialize()
    # Set master addr and port after jax distributed initialization
    if MASTER_ADDR:
        os.environ['MASTER_ADDR'] = MASTER_ADDR
    if MASTER_PORT:
        os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    init_logging()
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
        distributed_instruction_ratio=1.0,
        custon_normalization_path=custon_normalization_path,
    )

    # Update norm_stats to data_config
    #     min_range = np.array([-0.1, -0.5, -0.5, -3.0, -3.0, -3.0 , -9, -0.1, -0.5, -0.5, -3.0, -3.0, -3.0 , -9], dtype=np.float32)
    #     max_range = np.array([0.5,  0.5,  0.5, 3.0, 3.0, 3.0, 9,0.5,  0.5,  0.5, 3.0, 3.0, 3.0, 9], dtype=np.float32)
    #     # max_min = [0.6, 1.0, 1.0, 6.0, 6.0, 6.0, 18, 0.6, 1.0, 1.0, 6.0, 6.0, 6.0, 18]
    #     action_stats = {
    #         'state_mean': min_range,
    #         'state_std': max_range - min_range,
    #         'action_mean': min_range,
    #         'action_std': max_range - min_range,
    #     }
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
            norm_stats = json.load(f)
            norm_stats['low_quantile'] = np.array(norm_stats['low_quantile'])
            norm_stats['high_quantile'] = np.array(norm_stats['high_quantile'])

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
        # buffer_size=300,
        device='jax',
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
        print(f"rank {jax.process_index()}: All processes synchronized after dataset initialization", flush=True)
    
    train_dataloader = dataset.get_train_dataloader()
    # iterator = iter(train_dataloader)
    # data = next(iterator)
    # assert False, f"data: {data[0].keys()}"


    rng = jax.random.key(cfg.training.seed)
    train_rng, init_rng = jax.random.split(rng)
    resume_train = default(cfg, "training.resume_train", False)

    mesh = sharding.make_mesh(cfg.training.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Ensure checkpoint directory exists before initializing checkpoint manager
    checkpoint_dir = config.checkpoint_dir
    if jax.process_index() == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created checkpoint directory: {checkpoint_dir}")

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=resume_train,
    )

    if jax.process_index() == 0:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    # else:

    logging.info(f"debug line 1")
    logging.info(f"train_dataloader: {type(train_dataloader)}")
    data_iter = iter(train_dataloader)
    logging.info(f"debug line 2")
    
    per_process_batch = next(data_iter)
    logging.info(f"debug line 3")
    per_process_batch[0] = convert_per_process_batch_to_jax(per_process_batch[0], data_sharding) # on cpu
    per_process_batch[1] = convert_per_process_batch_to_jax(per_process_batch[1], data_sharding) # on cpu
    # logging.info(f"Initialized before:\n{training_utils.array_tree_to_info(per_process_batch)}")
    logging.info(f"debug line 4")
    batch = Observation.from_dict(per_process_batch[0]), per_process_batch[1]
    local_shape = per_process_batch[1].sharding.shard_shape(per_process_batch[1].shape)
    logging.info(f"Local shape: {local_shape}")
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    # assert False

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state)
        logging.info(f"Restored train state: from {checkpoint_manager.directory}")
        # assert False, "debug line 5" # TODO: Test resume train

    logging.info(f"Jit compiling train_step")
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    epoch = 0
    try:
        for step in pbar:
            with sharding.set_mesh(mesh):
                try:
                    train_state, info = ptrain_step(train_rng, train_state, batch)
                except Exception as e:
                    import traceback
                    full_traceback = traceback.format_exc()
                    logging.error(f"Error in training step: {e}\n{full_traceback}")
                    raise  # This will cause process to exit with error
                infos.append(info)
                if step % config.log_interval == 0:
                    stacked_infos = common_utils.stack_forest(infos)
                    reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                    info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                    pbar.write(f"Step {step}: {info_str}")
                    if jax.process_index() == 0:
                        wandb.log(reduced_info, step=step)
                    infos = []
                
                try:
                    per_process_batch = next(data_iter)
                except:
                    # If dataset is exhausted, reinitilize the dataloader
                    train_dataloader = dataset.get_train_dataloader()
                    data_iter = iter(train_dataloader)
                    per_process_batch = next(data_iter)
                per_process_batch[0] = convert_per_process_batch_to_jax(per_process_batch[0], data_sharding) # on cpu
                per_process_batch[1] = convert_per_process_batch_to_jax(per_process_batch[1], data_sharding) # on cpu
                batch = Observation.from_dict(per_process_batch[0]), per_process_batch[1]

                # Checkpoint saving logic
                if step % 5000 == 0 and step != 0:
                    # Synchronize all processes before checkpoint saving
                    multihost_utils.sync_global_devices("Before checkpoint saving")
                    logging.info(f"Saving checkpoint at step {step}")
                    _checkpoints.save_custom_state(checkpoint_manager, train_state, step)
                    # Synchronize all processes after checkpoint saving
                    multihost_utils.sync_global_devices("After checkpoint saving")
                    logging.info(f"Checkpoint saved at step {step}")

    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        logging.error(f"Process {jax.process_index()} failed with error: {e}\n{full_traceback}")
        # Exit with error code
        os._exit(1)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
