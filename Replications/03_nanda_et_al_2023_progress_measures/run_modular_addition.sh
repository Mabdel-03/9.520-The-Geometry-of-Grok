#!/bin/bash
#SBATCH --job-name=grok_nanda_modadd
#SBATCH --output=logs/modular_addition_%j.out
#SBATCH --error=logs/modular_addition_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Progress Measures for Grokking via Mechanistic Interpretability
# Nanda et al. (2023)

mkdir -p logs checkpoints

# Load modules
module load python/3.9
module load cuda/11.8

cd $SLURM_SUBMIT_DIR

# Install dependencies
pip install torch numpy matplotlib --quiet

# Run modular addition with 1-layer ReLU Transformer
# P=113, 30% training data
python train.py \
    --p=113 \
    --train_fraction=0.3 \
    --d_model=128 \
    --n_heads=4 \
    --d_mlp=512 \
    --lr=0.001 \
    --weight_decay=1.0 \
    --n_epochs=40000 \
    --log_interval=100 \
    --device=cuda \
    --seed=42

echo "Training complete!"

