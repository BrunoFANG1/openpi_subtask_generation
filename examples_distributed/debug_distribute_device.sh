#!/bin/bash
# Step 1: salloc --nodes=2 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=12 --mem=175G --exclude master 
# Step 2: srun bash srun_script.sh


export NCCL_BUFFSIZE=4194304  # 4 MiB = 4 * 1024 * 1024 bytes
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NVLS_ENABLE=0
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat
export TF_CPP_VMODULE=asm_compiler=5 
export TF_CPP_MIN_LOG_LEVEL=0
export TF_XLA_FLAGS="--tf_xla_dump_hlo_graphs"
export TF_CPP_MIN_LOG_LEVEL=0  # Ensure all logs are visible

set -e

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
# export NCCL_DEBUG=INFO
export JAX_PROCESS_COUNT=$SLURM_NTASKS
export JAX_PROCESS_INDEX=$SLURM_PROCID
export JAX_LOCAL_PROCESS_INDEX=$SLURM_LOCALID
export JAX_NODE_RANK=$SLURM_NODEID
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=12355

# Function to check CUDA version
check_cuda_version() {
    # Get CUDA version from nvidia-smi
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi command not found. Is NVIDIA driver installed?"
        return 1
    fi
    
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "Detected CUDA Version: $CUDA_VERSION"
    
    # Check if version is 12.8
    if [[ "$CUDA_VERSION" != "12.8" ]]; then
        echo "ERROR: CUDA version is not 12.8 (detected: $CUDA_VERSION)"
        echo "Host $(hostname): CUDA version mismatch"
        
        # Check if CUDA 12.8 is installed but not active
        if [ -d "/usr/local/cuda-12.8" ]; then
            echo "CUDA 12.8 appears to be installed at /usr/local/cuda-12.8"
            echo "You may need to update your environment variables:"
            echo "  export PATH=/usr/local/cuda-12.8/bin:\$PATH"
            echo "  export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH"
        else
            echo "CUDA 12.8 does not appear to be installed"
            echo "Please install CUDA 12.8 or update your CUDA installation"
        fi
        
        return 1
    else
        echo "CUDA 12.8 verification successful"
        return 0
    fi
}

# Main execution
echo "Checking CUDA version..."
if check_cuda_version; then
    echo "CUDA version check passed. Proceeding with execution."
else
    echo "CUDA version check failed. Exiting."
    exit 1
fi

# ── Launch your training ──
cd $(pwd)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run /x2robot/xinyuanfang/projects/openpi/test/test_distributed_device.py