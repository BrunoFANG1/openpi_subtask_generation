#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

export http_proxy=http://192.168.1.36:3128
export https_proxy=http://192.168.1.36:3128

set -e

which python
echo "Starting single-GPU test run..."
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export MASTER_ADDR=localhost
export MASTER_PORT=12356

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train_x2robot.py --config-name tuili_distribute