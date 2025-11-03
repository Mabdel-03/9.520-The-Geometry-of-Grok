#!/bin/bash
#SBATCH --job-name=grok_poly_add
#SBATCH --output=logs/modular_addition_%j.out
#SBATCH --error=logs/modular_addition_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Grokking Modular Polynomials
# Doshi et al. (2024)

mkdir -p logs checkpoints

# Load modules
module load python/3.9
module load cuda/11.8

cd $SLURM_SUBMIT_DIR

# Install dependencies
pip install torch numpy matplotlib --quiet

# Run modular addition with power activation MLP
# Architecture: 2-layer MLP with phi(x) = x^power
python train.py \
    --task=addition \
    --p=97 \
    --num_terms=2 \
    --hidden_dim=500 \
    --power=2 \
    --train_fraction=0.5 \
    --lr=0.005 \
    --weight_decay=5.0 \
    --n_epochs=50000 \
    --log_interval=100 \
    --device=cuda \
    --seed=42

echo "Modular polynomial training complete!"

