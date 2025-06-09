#!/bin/bash
#SBATCH --job-name=openpi_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:0
#SBATCH --mem=1200G
#SBATCH -o /x2robot/brae/projects/openpi/slurm_scripts/%x-%j.log

uv run examples/x2robot/convert_x2robot_data_to_lerobot.py --data_dir ""