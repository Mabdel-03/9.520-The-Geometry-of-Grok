#!/bin/bash
#SBATCH --job-name=softmax_agop
#SBATCH --output=logs/softmax_agop_%A_%a.out
#SBATCH --error=logs/softmax_agop_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GEFORCERTX2080:1
#SBATCH --partition=normal

# Softmax modular addition - FULL SWEEP
# Architectures: MLP, Transformer (2)
# Optimizers: adamw, muon, sgd (3)
# Weight decays: 0.01, 0.1, 0.5, 1.0 (4)
# Total: 2 × 3 × 4 = 24 jobs

CONDA_ENV=/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

mkdir -p logs

ARCHITECTURES=("mlp" "transformer")
OPTIMIZERS=("adamw" "muon" "sgd")
WEIGHT_DECAYS=(0.01 0.1 0.5 1.0)

ARCH_IDX=$((SLURM_ARRAY_TASK_ID / 12))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 12))
OPT_IDX=$((REMAINDER / 4))
WD_IDX=$((REMAINDER % 4))

ARCHITECTURE=${ARCHITECTURES[$ARCH_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_IDX]}

echo "Running Softmax AGOP experiment - FULL SWEEP"
echo "Architecture: $ARCHITECTURE"
echo "Optimizer: $OPTIMIZER"
echo "Weight Decay: $WEIGHT_DECAY"

$CONDA_ENV/bin/python ../training_scripts/train_softmax_agop.py \
    --architecture $ARCHITECTURE \
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
    --save_dir ../results/softmax \
    --experiment_name softmax_${ARCHITECTURE}_${OPTIMIZER}_wd${WEIGHT_DECAY}_seed42

echo "Job completed"


