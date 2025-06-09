import os
os.environ["HF_LEROBOT_HOME"] = "/x2robot/brae/.cache/lerobot"
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['OPENPI_DATA_HOME'] = '/x2robot/brae/.cache/openpi'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gc
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


def sigterm_handler(signum, frame):
    logging.info(f"Process {jax.process_index()} received SIGTERM, exiting")
    os._exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)


def init_logging():
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
        logger.setLevel(logging.INFO)
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
            mode='offline'
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


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    # from openpi.training.config import get_config

    # Due to type checking in openpi code, we need to get TrainConfig from get_config or manually setup one
    # try:
    #     config = get_config(cfg.config_name)
    #     logging.info(f"Using default config from openpi: {config}")
    # except:
    config = _config.TrainConfig(
        name=cfg.config_name,
        exp_name=cfg.exp_name,
        model=instantiate(cfg.model.model_config),  # Use Hydra's instantiate to create Pi0Config instance
        weight_loader=instantiate(cfg.model.weight_loader),
        data=instantiate(cfg.model.data),
        num_train_steps=cfg.model.num_train_steps,
    )

    if int(os.environ.get("SLURM_NTASKS", "0")) > 1:
        jax.distributed.initialize()
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("/x2robot/brae/.cache/openpi/jax").expanduser()))

    # Create dataloader
    train_dataloader, val_dataloader = _data_loader.create_x2robot_dataloader(cfg, jax_process_id=jax.process_index(), collate_type=cfg.collate_type)

    rng = jax.random.key(cfg.training.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(cfg.training.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=True,
        resume=False,
    )

    if jax.process_index() == 0:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    # else:
    resuming = False

    data_iter = iter(train_dataloader)
    per_process_batch = next(data_iter)
    per_process_batch[0] = convert_per_process_batch_to_jax(per_process_batch[0], data_sharding) # on cpu
    per_process_batch[1] = convert_per_process_batch_to_jax(per_process_batch[1], data_sharding) # on cpu
    # logging.info(f"Initialized before:\n{training_utils.array_tree_to_info(per_process_batch)}")
    batch = Observation.from_dict(per_process_batch[0]), per_process_batch[1]
    local_shape = per_process_batch[1].sharding.shard_shape(per_process_batch[1].shape)
    logging.info(f"Local shape: {local_shape}")
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    # assert False

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        pass
        # train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

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
    # try:
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
            
            # Try to get the next batch
            iterator_exhausted = False
            try:
                per_process_batch = next(data_iter)
            except StopIteration:
                iterator_exhausted = True
            per_process_iterator_exhausted_value = np.array(1.0 if iterator_exhausted else 0.0)
            global_iterator_exhausted_value = multihost_utils.process_allgather(per_process_iterator_exhausted_value)
            iterator_exhausted_count = jnp.sum(global_iterator_exhausted_value)

            # If any process's iterator is exhausted, all processes should reinitialize the data iterator
            if iterator_exhausted_count > 0:
                logging.info(f"Global iterator_exhausted_value: {global_iterator_exhausted_value}")
                logging.info(f"Process {jax.process_index()}: Dataset exhausted, reinitializing data iterator")
                gc.collect()
                train_dataloader.shutdown()
                epoch += 1
                train_dataloader.dataset.reset_epoch(epoch)
                data_iter = iter(train_dataloader)
                per_process_batch = next(data_iter)
            
            per_process_batch[0] = convert_per_process_batch_to_jax(per_process_batch[0], data_sharding) # on cpu
            per_process_batch[1] = convert_per_process_batch_to_jax(per_process_batch[1], data_sharding) # on cpu
            batch = Observation.from_dict(per_process_batch[0]), per_process_batch[1]

            # Checkpoint saving logic
            if step % 20000 == 0 and step != 0:
                logging.info(f"Saving checkpoint at step {step}")
                _checkpoints.save_custom_state(checkpoint_manager, train_state, step)
                logging.info(f"Checkpoint saved at step {step}")

    # except Exception as e:
    #     import traceback
    #     full_traceback = traceback.format_exc()
    #     logging.error(f"Process {jax.process_index()} failed with error: {e}\n{full_traceback}")
    #     # Exit with error code
    #     os._exit(1)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
