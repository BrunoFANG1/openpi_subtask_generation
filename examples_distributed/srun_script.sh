#!/bin/bash
# Step 1: salloc --nodes=2 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=12 --mem=175G --exclude master 
# Step 2: srun bash srun_script.sh

# Debug info
echo "Running on host: $(hostname)"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "SLURM_LOCALID=$SLURM_LOCALID"
echo "SLURM_NODEID=$SLURM_NODEID"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# (No need to manually remap CUDA_VISIBLE_DEVICES)

# ── JAX distributed env ──
export JAX_USE_PJRT_CUDA_DEVICE=True
export NCCL_DEBUG=INFO
export JAX_PROCESS_COUNT=$SLURM_NTASKS
export JAX_PROCESS_INDEX=$SLURM_PROCID
export JAX_LOCAL_PROCESS_INDEX=$SLURM_LOCALID
export JAX_NODE_RANK=$SLURM_NODEID

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=12355

# ── Launch your training ──
cd $(pwd)
uv run /x2robot/xinyuanfang/projects/openpi/test_jax.py
