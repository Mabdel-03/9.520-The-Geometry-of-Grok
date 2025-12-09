#!/bin/bash
#SBATCH --job-name=nanda_wd
#SBATCH --output=logs/nanda_wd_%A_%a.out
#SBATCH --error=logs/nanda_wd_%A_%a.err
#SBATCH --array=0-53
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --constraint="turing|ampere"

# Nanda Modular Addition - Extended Weight Decay Sweep
# Architectures: MLP, Transformer (2)
# Optimizers: adamw, muon, sgd (3)
# Weight decays: 0, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100 (9)
# Total: 2 × 3 × 9 = 54 jobs

# Activate conda environment
CONDA_ENV=/om/scratch/Mon/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

mkdir -p logs

ARCHITECTURES=("mlp" "transformer")
OPTIMIZERS=("adamw" "muon" "sgd")
WEIGHT_DECAYS=("0" "0.01" "0.1" "0.5" "1.0" "5.0" "10.0" "50.0" "100.0")

# Calculate indices: arch_idx * 27 + opt_idx * 9 + wd_idx
ARCH_IDX=$((SLURM_ARRAY_TASK_ID / 27))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 27))
OPT_IDX=$((REMAINDER / 9))
WD_IDX=$((REMAINDER % 9))

ARCHITECTURE=${ARCHITECTURES[$ARCH_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_IDX]}

echo "Running Nanda Extended WD Sweep"
echo "Architecture: $ARCHITECTURE"
echo "Optimizer: $OPTIMIZER"
echo "Weight Decay: $WEIGHT_DECAY"

cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments/slurm_scripts

# Run training with AGOP + Lazy-Rich tracking
$CONDA_ENV/bin/python ../training_scripts/train_nanda_agop.py \
    --architecture $ARCHITECTURE \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --n_epochs 40000 \
    --lr 0.001 \
    --train_fraction 0.3 \
    --agop_freq 100 \
    --agop_top_k 20 \
    --ntk_subsample 200 \
    --seed 42 \
    --save_dir ../results/nanda_wd_sweep

echo "Job completed"

