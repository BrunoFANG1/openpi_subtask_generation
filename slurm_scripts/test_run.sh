#!/bin/bash
#SBATCH --job-name=pi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=124
#SBATCH --gres=gpu:8
#SBATCH --mem=1600G
#SBATCH --exclude=master
#SBATCH -o /x2robot/xinyuanfang/projects/openpi/slurm_scripts/%x-%j.log

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_x2robot_office_sort_and_fold_clothes --exp-name=pi0_x2robot_sort_and_fold_clothes --overwrite