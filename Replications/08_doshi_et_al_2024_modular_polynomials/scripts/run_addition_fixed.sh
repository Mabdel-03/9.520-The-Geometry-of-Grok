#!/bin/bash
#SBATCH --job-name=grok_poly_fixed
#SBATCH --output=logs/polynomial_fixed_%j.out
#SBATCH --error=logs/polynomial_fixed_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Modular Polynomials with Power Activation - FIXED VERSION
# Doshi et al. (2024)
# Fixed: Import error, better hyperparameters

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
echo "Paper 08: Modular Polynomials (FIXED)"
echo "=========================================="
echo "Working directory: $(pwd)"
ls train.py || { echo "ERROR: train.py not found!"; exit 1; }

# Run with fixed configuration
# Key fixes:
# - Fixed F.one_hot import error in model.py
# - Adjusted learning rate (0.005 -> 0.001)
# - Adjusted weight decay (5.0 -> 1.0)
# - Using 2 terms (simpler task)
# - 100K epochs for faster iteration

python train.py \
    --task=addition \
    --p=97 \
    --num_terms=2 \
    --hidden_dim=500 \
    --power=2 \
    --train_fraction=0.5 \
    --lr=0.001 \
    --weight_decay=1.0 \
    --n_epochs=100000 \
    --log_interval=100 \
    --device=cuda \
    --seed=42

echo "=========================================="
echo "Modular polynomial experiment complete!"
echo "Check logs/training_history.json"
echo "=========================================="

