#!/bin/bash
#SBATCH --job-name=grok_power_modadd
#SBATCH --partition=use-everything
#SBATCH --output=modular_addition_%j.out
#SBATCH --error=modular_addition_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# OpenAI Grok - Modular Addition Experiment
# Power et al. (2022)

# Create logs directory
mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to the replication directory
cd $SLURM_SUBMIT_DIR

# Install package if not already installed
pip install -e . --quiet 2>/dev/null || echo "Package already installed"

# Run training for modular addition
# Default parameters from paper:
# - Operation: x + y (mod 97)
# - Training fraction: 50%
# - Weight decay: 1
# - Learning rate: 1e-3
# - Optimizer: AdamW

python scripts/train.py \
    --math_operator=x+y \
    --train_data_pct=0.5 \
    --weight_decay=1.0 \
    --max_lr=1e-3 \
    --max_steps=100000 \
    --save_activations \
    --save_outputs

echo "Training complete!"

