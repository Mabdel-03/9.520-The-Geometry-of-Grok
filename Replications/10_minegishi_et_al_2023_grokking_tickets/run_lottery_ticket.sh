#!/bin/bash
#SBATCH --job-name=grok_lottery
#SBATCH --output=lottery_ticket_%j.out
#SBATCH --error=lottery_ticket_%j.err
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

# Ensure we're in the right directory
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR="$(pwd)"
fi
cd "$SLURM_SUBMIT_DIR"

echo "Working directory: $(pwd)"
ls train.py || { echo "ERROR: train.py not found!"; exit 1; }

# Install required packages
pip install wandb torcheval einops --quiet 2>/dev/null || echo "Dependencies check"

# Run lottery ticket experiment on modular addition
# Train with pruning to observe accelerated grokking
python train.py \
    --task modular_addition \
    --seed 42 \
    --epochs 100000 \
    --log_interval 1000

echo "Lottery ticket experiment complete!"

