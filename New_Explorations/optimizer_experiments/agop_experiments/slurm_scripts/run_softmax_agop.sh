#!/bin/bash
#SBATCH --job-name=softmax_agop
#SBATCH --output=logs/softmax_agop_%A_%a.out
#SBATCH --error=logs/softmax_agop_%A_%a.err
#SBATCH --array=0-11
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

# Softmax transformer modular addition with AGOP tracking
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

echo "Running Softmax AGOP experiment"
echo "Optimizer: $OPTIMIZER"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Job ID: $SLURM_JOB_ID"

$CONDA_ENV/bin/python ../train_softmax_agop.py \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --lr 0.001 \
    --p 97 \
    --train_fraction 0.5 \
    --n_epochs 50000 \
    --agop_freq 100 \
    --agop_top_k 20 \
    --log_freq 100 \
    --device cuda \
    --seed 42 \
    --save_dir ./results/agop_experiments/softmax \
    --experiment_name softmax_${OPTIMIZER}_wd${WEIGHT_DECAY}_seed42

echo "Job completed"

