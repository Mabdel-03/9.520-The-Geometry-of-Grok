#!/bin/bash
#SBATCH --job-name=grok_power_modadd
#SBATCH --output=logs/modular_addition_%j.out
#SBATCH --error=logs/modular_addition_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# OpenAI Grok - Modular Addition Experiment
# Power et al. (2022)

# Create logs directory
mkdir -p logs

# Load modules (adjust based on your HPC system)
module load python/3.9
module load cuda/11.8

# Activate virtual environment (adjust path as needed)
# source ../../venv/bin/activate

# Navigate to the replication directory
cd $SLURM_SUBMIT_DIR

# Install package if not already installed
pip install -e . --quiet

# Run training for modular addition
# Default parameters from paper:
# - Operation: x + y (mod 97)
# - Training fraction: 50%
# - Weight decay: 1
# - Learning rate: 1e-3
# - Optimizer: AdamW

python scripts/train.py \
    --operation=x+y \
    --prime=97 \
    --training_fraction=0.5 \
    --weight_decay=1 \
    --learning_rate=1e-3 \
    --optimizer=adamw \
    --steps=100000 \
    --save_every=1000

echo "Training complete!"

