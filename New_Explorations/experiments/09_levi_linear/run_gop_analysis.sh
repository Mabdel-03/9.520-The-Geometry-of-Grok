#!/bin/bash
#SBATCH --job-name=gop_linear
#SBATCH --partition=use-everything
#SBATCH --output=gop_linear_%j.out
#SBATCH --error=gop_linear_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# GOP Analysis for Linear Estimators (smallest model - good for testing)
set -e  # Exit on error
set -x  # Print commands

# Create logs directory
mkdir -p logs

echo "Starting GOP analysis..."
echo "Current directory: $(pwd)"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# Activate conda environment
echo "Activating conda environment..."
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

echo "Conda activated. Python location: $(which python)"
python --version

# Navigate to submit directory
cd $SLURM_SUBMIT_DIR
echo "Changed to directory: $(pwd)"

# Run wrapped training
echo "Starting training..."
python wrapped_train.py --config ../../configs/09_levi_linear.yaml

echo "GOP analysis complete!"

