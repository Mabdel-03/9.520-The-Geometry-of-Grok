#!/bin/bash
#SBATCH --job-name=grok_slingshot_fixed
#SBATCH --output=logs/slingshot_fixed_%j.out
#SBATCH --error=logs/slingshot_fixed_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# The Slingshot Mechanism - FIXED VERSION
# Thilak et al. (2022)
# Fixed: Better initialization, AdamW optimizer, proper weight decay

mkdir -p logs checkpoints

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Ensure we're in the right directory
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR="$(pwd)"
fi
cd "$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "Paper 07: Slingshot Mechanism (FIXED)"
echo "=========================================="
echo "Working directory: $(pwd)"
ls train.py || { echo "ERROR: train.py not found!"; exit 1; }

# Run with fixed configuration
# Key changes from previous run:
# - Using AdamW instead of Adam
# - Reduced epochs to 100K for faster iteration
# - Better model initialization (added in model.py)
# - Smaller modulus (p=97 instead of 113)

python train.py \
    --p=97 \
    --train_fraction=0.5 \
    --d_model=128 \
    --n_heads=4 \
    --n_layers=2 \
    --d_mlp=512 \
    --optimizer=adamw \
    --lr=0.001 \
    --weight_decay=1.0 \
    --n_epochs=100000 \
    --log_interval=100 \
    --device=cuda \
    --seed=42

echo "=========================================="
echo "Slingshot experiment complete!"
echo "Check logs/training_history.json"
echo "=========================================="

