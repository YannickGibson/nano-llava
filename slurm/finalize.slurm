#!/bin/bash
#SBATCH --job-name=nllava-fin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Post-training: render the qualitative grid and run the VQAv2 ablation.
#   sbatch -p <gpu-partition> slurm/finalize.slurm

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

uv run python grid.py --out assets/example_grid.png
uv run python ablation.py --n 2000
