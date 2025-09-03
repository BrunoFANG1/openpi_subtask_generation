import os
os.environ["HF_LEROBOT_HOME"] = "/x2robot_v2/xinyuanfang/projects_v2/.cache/lerobot"
# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['OPENPI_DATA_HOME'] = '/x2robot_v2/xinyuanfang/projects_v2/.cache/openpi'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)
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
    real_action_dim: int,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, real_action_dim=real_action_dim, train=True)
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

    init_logging(cfg.training.debug)
    logging.info(f"Running on: {platform.node()}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("/x2robot_v2/xinyuanfang/projects_v2/.cache/openpi/jax").expanduser()))

    rng = jax.random.key(cfg.training.seed)
    train_rng, init_rng = jax.random.split(rng)
    resume_train = default(cfg, "training.resume_train", False)

    mesh = sharding.make_mesh(cfg.training.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Create dataloader
    dataset, train_dataloader, val_dataloader = _data_loader.create_x2robot_dataloader(cfg)
    train_dataloader.data_sharding = data_sharding
    val_dataloader.data_sharding = data_sharding # TODO: find more elegant way

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
    
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    # assert False

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, mesh=mesh)
        logging.info(f"Restored train state: from {checkpoint_manager.directory}")
        # assert False, "debug line 5" # TODO: Test resume train

    logging.info(f"Jit compiling train_step")
    ptrain_step = jax.jit(
        functools.partial(train_step, config, real_action_dim=cfg.task.shape_meta.action.shape[0]),
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
                train_state, info = ptrain_step(train_rng, train_state, batch)
                infos.append(info)
                if step % config.log_interval == 0:
                    stacked_infos = common_utils.stack_forest(infos)
                    reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                    epoch_percentage = step / dataset.global_train_iters.value
                    info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
                    pbar.write(f"Step {step}: {info_str}")
                    if jax.process_index() == 0:
                        wandb.log(reduced_info, step=step)
                    infos = []
                
                try:
                    batch = next(data_iter)
                except:
                    # If dataset is exhausted, reinitilize the dataloader
                    dataset, train_dataloader, val_dataloader = _data_loader.create_x2robot_dataloader(cfg)
                    train_dataloader.data_sharding = data_sharding
                    val_dataloader.data_sharding = data_sharding # TODO: find more elegant way
                    epoch += 1
                    data_iter = iter(train_dataloader)
                    batch = next(data_iter)

                # Checkpoint saving logic
                if step % cfg.checkpoint.keep_period == 0 and step != 0:
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
