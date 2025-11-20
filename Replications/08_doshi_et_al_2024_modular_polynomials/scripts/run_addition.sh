#!/bin/bash
#SBATCH --job-name=grok_poly_add
#SBATCH --output=modular_addition_%j.out
#SBATCH --error=modular_addition_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Grokking Modular Polynomials
# Doshi et al. (2024)
# NOTE: Extended epochs and adjusted LR for grokking

mkdir -p logs checkpoints

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Ensure we're in the right directory
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR="$(pwd)"
fi
cd "$SLURM_SUBMIT_DIR"

echo "Working directory: $(pwd)"
ls train.py || { echo "ERROR: train.py not found!"; exit 1; }

# Run modular addition with power activation MLP
# Architecture: 2-layer MLP with phi(x) = x^power
# Reduced LR and increased epochs for better grokking
python train.py \
    --task=addition \
    --p=97 \
    --num_terms=2 \
    --hidden_dim=500 \
    --power=2 \
    --train_fraction=0.5 \
    --lr=0.001 \
    --weight_decay=3.0 \
    --n_epochs=200000 \
    --log_interval=500 \
    --device=cuda \
    --seed=42

echo "Modular polynomial training complete!"

