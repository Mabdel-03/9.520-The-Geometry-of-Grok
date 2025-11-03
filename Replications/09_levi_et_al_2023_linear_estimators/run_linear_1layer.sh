#!/bin/bash
#SBATCH --job-name=grok_linear_1layer
#SBATCH --output=logs/linear_1layer_%j.out
#SBATCH --error=logs/linear_1layer_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2

# Grokking in Linear Estimators
# Levi et al. (2023)

mkdir -p logs checkpoints

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

cd $SLURM_SUBMIT_DIR

# Run 1-layer linear teacher-student
# Demonstrates grokking in a solvable linear model
python train.py \
    --architecture=1layer \
    --d_in=1000 \
    --d_out=1 \
    --n_train=500 \
    --n_test=10000 \
    --lr=0.01 \
    --weight_decay=0.01 \
    --n_epochs=100000 \
    --log_interval=100 \
    --accuracy_threshold=1e-3 \
    --device=cuda \
    --seed=42

echo "Linear estimator training complete!"

