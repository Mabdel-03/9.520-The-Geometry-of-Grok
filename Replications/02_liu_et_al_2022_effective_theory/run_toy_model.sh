#!/bin/bash
#SBATCH --job-name=grok_liu_toy
#SBATCH --partition=use-everything
#SBATCH --output=toy_model_%j.out
#SBATCH --error=toy_model_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2

# Effective Theory of Representation Learning
# Liu et al. (2022)

mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Ensure we're in the right directory
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR="$(pwd)"
fi
cd "$SLURM_SUBMIT_DIR"

echo "Working directory: $(pwd)"
ls scripts/run_toy_model.py || { echo "ERROR: scripts/run_toy_model.py not found!"; exit 1; }

# Install dependencies if needed
pip install sacred pymongo --quiet 2>/dev/null || echo "Dependencies already installed"
pip install -r requirements.txt --quiet 2>/dev/null || echo "Requirements file not found"

# Run toy model experiment  
# Demonstrates phase diagram and grokking transitions
# Sacred uses "with" syntax with correct parameter names

python scripts/run_toy_model.py with \
    p=10 \
    train_fraction=0.45 \
    train_steps=50000 \
    encoder_lr=1e-3 \
    decoder_lr=1e-3 \
    encoder_weight_decay=1.0 \
    decoder_weight_decay=1.0 \
    seed=0

echo "Toy model training complete!"

