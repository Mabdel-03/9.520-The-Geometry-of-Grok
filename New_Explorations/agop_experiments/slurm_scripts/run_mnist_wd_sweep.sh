#!/bin/bash
#SBATCH --job-name=mnist_wd
#SBATCH --output=logs/mnist_wd_%A_%a.out
#SBATCH --error=logs/mnist_wd_%A_%a.err
#SBATCH --array=0-26
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --constraint="turing|ampere"

# MNIST MLP - Extended Weight Decay Sweep
# Optimizers: adamw, muon, sgd (3)
# Weight decays: 0, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100 (9)
# Total: 3 × 9 = 27 jobs

# Activate conda environment
CONDA_ENV=/om/scratch/Mon/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

mkdir -p logs

OPTIMIZERS=("adamw" "muon" "sgd")
WEIGHT_DECAYS=("0" "0.01" "0.1" "0.5" "1.0" "5.0" "10.0" "50.0" "100.0")

# Calculate indices
OPT_IDX=$((SLURM_ARRAY_TASK_ID / 9))
WD_IDX=$((SLURM_ARRAY_TASK_ID % 9))

OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_IDX]}

echo "Running MNIST Extended WD Sweep"
echo "Optimizer: $OPTIMIZER"
echo "Weight Decay: $WEIGHT_DECAY"

cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments/slurm_scripts

# Run training with AGOP + Lazy-Rich tracking
$CONDA_ENV/bin/python ../training_scripts/train_mnist_agop.py \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --n_epochs 10000 \
    --lr 0.001 \
    --hidden_dim 512 \
    --depth 3 \
    --train_points 1000 \
    --agop_freq 100 \
    --agop_subsample 500 \
    --agop_top_k 20 \
    --ntk_subsample 200 \
    --seed 42 \
    --save_dir ../results/mnist_wd_sweep

echo "Job completed"

