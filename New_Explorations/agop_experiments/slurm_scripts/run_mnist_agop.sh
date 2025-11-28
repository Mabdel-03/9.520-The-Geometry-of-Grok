#!/bin/bash
#SBATCH --job-name=mnist_agop
#SBATCH --output=logs/mnist_agop_%A_%a.out
#SBATCH --error=logs/mnist_agop_%A_%a.err
#SBATCH --array=0-11
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

# MNIST Omnigrok with AGOP tracking
# Array job for optimizer x weight_decay sweep

# Activate conda environment
CONDA_ENV=/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

mkdir -p logs

OPTIMIZERS=("adamw" "muon" "sgd")
WEIGHT_DECAYS=(0.01 0.1 0.5 1.0)

OPT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
WD_IDX=$((SLURM_ARRAY_TASK_ID % 4))

OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_IDX]}

echo "Running MNIST AGOP experiment"
echo "Optimizer: $OPTIMIZER"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Job ID: $SLURM_JOB_ID"

$CONDA_ENV/bin/python ../training_scripts/train_mnist_agop.py \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --lr 0.001 \
    --train_points 1000 \
    --hidden_dim 200 \
    --depth 3 \
    --n_epochs 50000 \
    --agop_freq 100 \
    --agop_subsample 500 \
    --agop_top_k 20 \
    --log_freq 100 \
    --device cuda \
    --seed 42 \
    --save_dir ./results/agop_experiments/mnist \
    --experiment_name mnist_${OPTIMIZER}_wd${WEIGHT_DECAY}_seed42

echo "Job completed"

