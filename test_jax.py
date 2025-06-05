import os
import jax
import time
from jax.experimental import multihost_utils
import jax.distributed

def main():
    jax.distributed.initialize()

    # Global rank (across all processes)
    global_rank = jax.process_index()
    world_size = jax.process_count()

    # Local rank (inside one node)
    # Use SLURM_LOCALID if available
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    local_world_size = int(os.environ.get('SLURM_NTASKS_PER_NODE', 8))  # default to 8 if missing

    print(f"python: Hello from host {jax.process_index()} / {jax.process_count()}")
    print(f"python: Global rank: {global_rank} / {world_size}")
    print(f"python: Local rank: {local_rank} / {local_world_size}")
    print(f"python: jax.device_count(): {jax.device_count()}")
    print(f"python: jax.local_device_count(): {jax.local_device_count()}")

    # print(f"Local devices: {jax.local_devices()}")
    # print(f"Global devices: {jax.devices()}")

if __name__ == "__main__":
    main()
