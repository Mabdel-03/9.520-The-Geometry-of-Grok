#!/bin/bash
#SBATCH --job-name=grok_lottery_fixed
#SBATCH --output=logs/lottery_fixed_%j.out
#SBATCH --error=logs/lottery_fixed_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Grokking Tickets: Lottery Tickets Accelerate Grokking - FIXED
# Minegishi et al. (2023)
# Fixed: Removed wandb dependency, saves to training_history.json

mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Ensure we're in the right directory
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR="$(pwd)"
fi
cd "$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "Paper 10: Lottery Tickets (FIXED)"
echo "=========================================="
echo "Working directory: $(pwd)"
ls train_no_wandb.py || { echo "ERROR: train_no_wandb.py not found!"; exit 1; }

# Run lottery ticket experiment without wandb
python train_no_wandb.py \
    --task modular_addition \
    --p 97 \
    --frac_train 0.5 \
    --epochs 100000 \
    --log_interval 100 \
    --seed 42

echo "=========================================="
echo "Lottery ticket experiment complete!"
echo "Check logs/training_history.json"
echo "=========================================="

