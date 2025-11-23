#!/bin/bash
#SBATCH --job-name=mnist_grok
#SBATCH --output=logs/mnist_%A_%a.out
#SBATCH --error=logs/mnist_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

# Single run of MNIST Omnigrok experiment
# Usage: sbatch run_mnist_single.sh <optimizer> <weight_decay>
# Example: sbatch run_mnist_single.sh adamw 0.01

OPTIMIZER=${1:-adamw}
WEIGHT_DECAY=${2:-0.01}

echo "=========================================="
echo "Paper 5: Liu et al. (2022) - MNIST Grokking"
echo "Optimizer: $OPTIMIZER"
echo "Weight Decay: $WEIGHT_DECAY"
echo "=========================================="

# Activate conda environment from scratch space
CONDA_ENV=/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

# Navigate to experiment directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/paper05_omnigrok

# Run training
$CONDA_ENV/bin/python train_mnist.py \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --lr 0.001 \
    --train_points 1000 \
    --hidden_dim 200 \
    --depth 3 \
    --activation relu \
    --init_scale 8.0 \
    --n_epochs 100000 \
    --batch_size 200 \
    --log_freq 100 \
    --checkpoint_freq 5000 \
    --spectral_freq 500 \
    --spectral_top_k 20 \
    --agop_subsample_size 500 \
    --save_dir ../results/paper05_omnigrok \
    --experiment_name "mnist_${OPTIMIZER}_wd${WEIGHT_DECAY}" \
    --device cuda \
    --seed 0

echo "=========================================="
echo "Job completed!"
echo "=========================================="

