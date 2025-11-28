#!/bin/bash
#SBATCH --job-name=nanda_grok
#SBATCH --output=logs/nanda_%A_%a.out
#SBATCH --error=logs/nanda_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Single run of Nanda et al. experiment
# Usage: sbatch run_nanda_single.sh <optimizer> <weight_decay>
# Example: sbatch run_nanda_single.sh adamw 1.0

OPTIMIZER=${1:-adamw}
WEIGHT_DECAY=${2:-1.0}

echo "=========================================="
echo "Paper 3: Nanda et al. (2023)"
echo "Optimizer: $OPTIMIZER"
echo "Weight Decay: $WEIGHT_DECAY"
echo "=========================================="

# Activate conda environment from scratch space
CONDA_ENV=/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

# Navigate to experiment directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/paper03_nanda

# Run training
$CONDA_ENV/bin/python train_nanda.py \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --lr 0.001 \
    --p 113 \
    --train_fraction 0.3 \
    --d_model 128 \
    --n_heads 4 \
    --d_mlp 512 \
    --n_epochs 40000 \
    --log_freq 100 \
    --checkpoint_freq 1000 \
    --spectral_freq 100 \
    --spectral_top_k 20 \
    --agop_subsample_size 1000 \
    --save_dir ../results/paper03_nanda \
    --experiment_name "nanda_${OPTIMIZER}_wd${WEIGHT_DECAY}" \
    --device cuda

echo "=========================================="
echo "Job completed!"
echo "=========================================="

