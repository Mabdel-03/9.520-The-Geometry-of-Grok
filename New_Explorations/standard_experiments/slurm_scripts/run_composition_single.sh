#!/bin/bash
#SBATCH --job-name=comp_grok
#SBATCH --output=logs/comp_%A_%a.out
#SBATCH --error=logs/comp_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

# Single run of Composition Reasoning experiment
# Usage: sbatch run_composition_single.sh <optimizer> <weight_decay>
# Example: sbatch run_composition_single.sh muonw 0.1

OPTIMIZER=${1:-adamw}
WEIGHT_DECAY=${2:-0.1}

echo "=========================================="
echo "Compositional Reasoning - GPT-2"
echo "Optimizer: $OPTIMIZER"
echo "Weight Decay: $WEIGHT_DECAY"
echo "=========================================="

# Conda environment from scratch space
CONDA_ENV=/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

# Navigate to experiment directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/paper04_composition

# Run training
$CONDA_ENV/bin/python train_composition.py \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --n_layer 4 \
    --n_embd 768 \
    --n_head 12 \
    --max_steps 150000 \
    --batch_size 512 \
    --log_freq 500 \
    --checkpoint_freq 10000 \
    --save_dir ../results/paper04_composition \
    --experiment_name "comp_${OPTIMIZER}_wd${WEIGHT_DECAY}" \
    --device cuda \
    --seed 42

echo "=========================================="
echo "Job completed!"
echo "=========================================="

