#!/bin/bash
#SBATCH --job-name=gop_nanda
#SBATCH --output=logs/gop_nanda_%j.out
#SBATCH --error=logs/gop_nanda_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

# GOP Analysis for Nanda et al. (2023)
# 1-layer ReLU Transformer on Modular Addition

# Create logs directory
mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to experiment directory
cd $SLURM_SUBMIT_DIR

# Run wrapped training with GOP tracking
python wrapped_train.py --config ../../configs/03_nanda_progress.yaml

echo "GOP analysis complete!"
echo "Check results in: ../../results/03_nanda_progress/"

