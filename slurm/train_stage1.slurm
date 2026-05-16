#!/bin/bash
#SBATCH --job-name=nllava-s1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Stage 1: projector pretraining on Flickr30k captions (~10 min on one A100).
#   sbatch -p <gpu-partition> slurm/train_stage1.slurm
#   sbatch -p <gpu-partition> slurm/train_stage1.slurm --wandb

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv run python train.py --stage 1 "$@"
