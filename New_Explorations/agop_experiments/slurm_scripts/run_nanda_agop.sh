#!/bin/bash
#SBATCH --job-name=nanda_agop
#SBATCH --output=logs/nanda_agop_%A_%a.out
#SBATCH --error=logs/nanda_agop_%A_%a.err
#SBATCH --array=0-11
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

# Nanda modular addition with AGOP tracking
# Array job for optimizer x weight_decay sweep

# Activate conda environment
CONDA_ENV=/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

# Create logs directory
mkdir -p logs

# Optimizers: adamw, muon, sgd (3 optimizers)
# Weight decays: 0.1, 1.0, 5.0, 10.0 (4 values)
# Total: 3 x 4 = 12 jobs

OPTIMIZERS=("adamw" "muon" "sgd")
WEIGHT_DECAYS=(0.1 1.0 5.0 10.0)

# Calculate optimizer and weight_decay from array index
OPT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
WD_IDX=$((SLURM_ARRAY_TASK_ID % 4))

OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_IDX]}

echo "Running Nanda AGOP experiment"
echo "Optimizer: $OPTIMIZER"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"

# Activate environment (adjust path as needed)
# source ~/miniconda3/bin/activate grokking

# Run training
$CONDA_ENV/bin/python ../training_scripts/train_nanda_agop.py \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --lr 0.001 \
    --p 113 \
    --train_fraction 0.3 \
    --n_epochs 40000 \
    --agop_freq 100 \
    --agop_top_k 20 \
    --log_freq 100 \
    --device cuda \
    --seed 42 \
    --save_dir ./results/agop_experiments/nanda \
    --experiment_name nanda_${OPTIMIZER}_wd${WEIGHT_DECAY}_seed42

echo "Job completed"

