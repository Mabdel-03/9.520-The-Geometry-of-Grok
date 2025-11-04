#!/bin/bash
#SBATCH --job-name=grok_paper01_logged
#SBATCH --partition=use-everything
#SBATCH --output=01_power_et_al_2022_openai_grok/logs/power_logged_%j.out
#SBATCH --error=01_power_et_al_2022_openai_grok/logs/power_logged_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

echo "=========================================="
echo "Paper 01: Power et al. (2022) - OpenAI Grok"
echo "With proper logging to training_history.json"
echo "=========================================="

# Create logs directory
mkdir -p 01_power_et_al_2022_openai_grok/logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/01_power_et_al_2022_openai_grok

# Install package if needed
pip install -e . --quiet 2>/dev/null || echo "Package already installed"

# Run with custom logging
python train_with_logging.py \
    --math_operator=x+y \
    --train_data_pct=0.5 \
    --weight_decay=1.0 \
    --max_lr=1e-3 \
    --max_steps=50000 \
    --val_every=500 \
    --logdir=./logs

echo "Training complete! Check logs/training_history.json"

