#!/bin/bash
#SBATCH --job-name=grok_effective_theory
#SBATCH --output=logs/effective_theory_%j.out
#SBATCH --error=logs/effective_theory_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2

# Effective Theory of Representation Learning - FIXED
# Liu et al. (2022)
# Fixed: Removed Sacred dependency, simple implementation

mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Ensure we're in the right directory
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR="$(pwd)"
fi
cd "$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "Paper 02: Effective Theory (FIXED)"
echo "=========================================="
echo "Working directory: $(pwd)"
ls train_simple.py || { echo "ERROR: train_simple.py not found!"; exit 1; }

# Run simplified toy model experiment
# Demonstrates grokking through representation learning
python train_simple.py \
    --p=10 \
    --d_hidden=200 \
    --train_fraction=0.45 \
    --train_steps=50000 \
    --encoder_lr=0.001 \
    --decoder_lr=0.001 \
    --encoder_wd=1.0 \
    --decoder_wd=1.0 \
    --log_interval=100 \
    --device=cuda \
    --seed=0

echo "=========================================="
echo "Effective theory experiment complete!"
echo "Check logs/training_history.json"
echo "=========================================="

