#!/bin/bash
#SBATCH --job-name=grok_paper02_logged
#SBATCH --partition=use-everything
#SBATCH --output=02_liu_et_al_2022_effective_theory/logs/toy_logged_%j.out
#SBATCH --error=02_liu_et_al_2022_effective_theory/logs/toy_logged_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2

echo "=========================================="
echo "Paper 02: Liu et al. (2022) - Effective Theory"
echo "With logging to training_history.json"
echo "=========================================="

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/02_liu_et_al_2022_effective_theory

# Run with logging
python run_toy_with_logging.py \
    --p=10 \
    --train_fraction=0.45 \
    --train_steps=50000 \
    --encoder_lr=1e-3 \
    --decoder_lr=1e-3 \
    --encoder_weight_decay=1.0 \
    --decoder_weight_decay=1.0 \
    --seed=0

echo "Done!"

