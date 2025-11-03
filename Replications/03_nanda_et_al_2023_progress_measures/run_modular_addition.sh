#!/bin/bash
#SBATCH --job-name=grok_nanda_modadd
#SBATCH --output=logs/modular_addition_%j.out
#SBATCH --error=logs/modular_addition_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Progress Measures for Grokking via Mechanistic Interpretability
# Nanda et al. (2023)

mkdir -p logs checkpoints

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

cd $SLURM_SUBMIT_DIR

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

