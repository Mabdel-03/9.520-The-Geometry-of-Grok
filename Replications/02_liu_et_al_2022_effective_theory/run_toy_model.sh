#!/bin/bash
#SBATCH --job-name=grok_liu_toy
#SBATCH --partition=use-everything
#SBATCH --output=logs/toy_model_%j.out
#SBATCH --error=logs/toy_model_%j.err
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

cd $SLURM_SUBMIT_DIR

# Install dependencies if needed
pip install -r requirements.txt --quiet 2>/dev/null || echo "Dependencies already installed"

# Run toy model experiment
# Demonstrates phase diagram and grokking transitions

python scripts/run_toy_model.py \
    --num_train=45 \
    --lr_embed=1e-3 \
    --lr_decoder=1e-3 \
    --weight_decay=1.0 \
    --epochs=50000 \
    --seed=0

echo "Toy model training complete!"

