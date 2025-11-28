#!/bin/bash
#SBATCH --job-name=softmax_grok
#SBATCH --output=logs/softmax_%A_%a.out
#SBATCH --error=logs/softmax_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Single run of Softmax Transformer modular arithmetic experiment
# Usage: sbatch run_softmax_single.sh <optimizer> <weight_decay>
# Example: sbatch run_softmax_single.sh muonw 0.1

OPTIMIZER=${1:-adamw}
WEIGHT_DECAY=${2:-0.1}

echo "=========================================="
echo "Modular Arithmetic - Softmax Transformer"
echo "Optimizer: $OPTIMIZER"
echo "Weight Decay: $WEIGHT_DECAY"
echo "=========================================="

# Conda environment from scratch space
CONDA_ENV=/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

# Navigate to experiment directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/paper03_softmax

# Run training
$CONDA_ENV/bin/python train_modular_softmax.py \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --p 97 \
    --train_fraction 0.5 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --d_ff 512 \
    --n_epochs 50000 \
    --log_freq 100 \
    --checkpoint_freq 1000 \
    --save_dir ../results/paper03_softmax \
    --experiment_name "softmax_${OPTIMIZER}_wd${WEIGHT_DECAY}" \
    --device cuda \
    --seed 0

echo "=========================================="
echo "Job completed!"
echo "=========================================="

