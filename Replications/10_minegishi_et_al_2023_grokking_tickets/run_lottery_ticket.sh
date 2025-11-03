#!/bin/bash
#SBATCH --job-name=grok_lottery
#SBATCH --output=logs/lottery_ticket_%j.out
#SBATCH --error=logs/lottery_ticket_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Grokking Tickets: Lottery Tickets Accelerate Grokking
# Minegishi et al. (2023)

mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

cd $SLURM_SUBMIT_DIR

# Run lottery ticket experiment
# Train -> prune -> retrain to observe accelerated grokking

python main.py \
    --dataset=modular_addition \
    --pruning_rate=0.6 \
    --seed=42

echo "Lottery ticket experiment complete!"

