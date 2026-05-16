#!/bin/bash
#SBATCH --job-name=nllava-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Run the VQAv2 ablation (stage-1-only vs stage-1+2) and write results.md.
#   sbatch -p <gpu-partition> slurm/eval.slurm
#   sbatch -p <gpu-partition> slurm/eval.slurm --n 1000

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
uv run python ablation.py "$@"
