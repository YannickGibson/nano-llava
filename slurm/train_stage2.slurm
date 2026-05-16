#!/bin/bash
#SBATCH --job-name=nllava-s2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Stage 2: visual instruction tuning with LoRA (~35 min on one A100).
# Requires checkpoints/stage1_projector.pt from stage 1.
#   sbatch -p <gpu-partition> slurm/train_stage2.slurm
#   sbatch -p <gpu-partition> slurm/train_stage2.slurm --wandb

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv run python train.py --stage 2 "$@"
