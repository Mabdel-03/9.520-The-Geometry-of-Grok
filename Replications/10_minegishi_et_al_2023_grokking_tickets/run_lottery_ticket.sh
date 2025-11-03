#!/bin/bash
#SBATCH --job-name=grok_lottery
#SBATCH --output=logs/lottery_ticket_%j.out
#SBATCH --error=logs/lottery_ticket_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Grokking Tickets: Lottery Tickets Accelerate Grokking
# Minegishi et al. (2023)

mkdir -p logs

# Load modules
module load python/3.9
module load cuda/11.8

cd $SLURM_SUBMIT_DIR

# Install dependencies
pip install -r requirements.txt --quiet 2>/dev/null || pip install torch numpy matplotlib --quiet

# Run lottery ticket experiment
# Train -> prune -> retrain to observe accelerated grokking

python main.py \
    --dataset=modular_addition \
    --pruning_rate=0.6 \
    --seed=42

echo "Lottery ticket experiment complete!"

