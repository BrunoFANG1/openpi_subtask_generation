#!/bin/bash
#SBATCH --job-name=pi0_fast
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=175G
#SBATCH --exclude=master
#SBATCH --output=/x2robot/xinyuanfang/projects/openpi/slurm_scripts/%x-%j.log

# Debug info
echo "Running on host: $(hostname)"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "SLURM_LOCALID=$SLURM_LOCALID"
echo "SLURM_NODEID=$SLURM_NODEID"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "Original CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# === Auto-select correct GPU ===
# Parse available GPUs
IFS=',' read -ra AVAILABLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_AVAILABLE_GPUS=${#AVAILABLE_GPUS[@]}

# Calculate which GPU to use
GPU_INDEX=$((SLURM_LOCALID % NUM_AVAILABLE_GPUS))

# Set correct CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${AVAILABLE_GPUS[$GPU_INDEX]}
echo "Mapped CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# === Setup JAX Distributed ===
export JAX_USE_PJRT_CUDA_DEVICE=True
export NCCL_DEBUG=INFO

export JAX_PROCESS_COUNT=$SLURM_NTASKS
export JAX_PROCESS_INDEX=$SLURM_PROCID
export JAX_LOCAL_PROCESS_INDEX=$SLURM_LOCALID
export JAX_NODE_RANK=$SLURM_NODEID

# Coordinator address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355

# 🚀 Launch the job
# 🚨 Start a persistent tmux session (only one copy!)
# if [ "$SLURM_LOCALID" -eq 0 ]; then
#     tmux new-session -d -s openpi_session "srun uv run test_jax.py; bash"
# fi
srun uv run test_jax.py


# 🚨 Wait until tmux exits
wait