#!/bin/bash
#SBATCH --job-name=icml_16runs
#SBATCH --output=logs/icml_16runs_%A_%a.out
#SBATCH --error=logs/icml_16runs_%A_%a.err
#SBATCH --array=0-95
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GEFORCERTX2080:1
#SBATCH --partition=normal

# ============================================================================
# ICML 16_Runs Experiment Sweep
# ============================================================================
#
# 16 base configurations × 6 weight decays = 96 total experiments
#
# Base configurations (16):
#   - Modulus: 97, 113 (2)
#   - Attention: softmax, relu (2)
#   - LayerNorm: on, off (2)
#   - Optimizer: adam, muon (2)
#
# Weight decays (6): 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0
#
# Array task mapping:
#   task_id = base_config_idx * 6 + weight_decay_idx
#   base_config_idx = task_id / 6
#   weight_decay_idx = task_id % 6
#
# ============================================================================

# Activate conda environment
CONDA_ENV=/om/scratch/Mon/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

# Create logs directory if needed
mkdir -p logs

# Define configuration arrays
MODULI=(97 113)
ATTENTION_TYPES=("softmax" "relu")
LAYERNORM_FLAGS=("--use_layernorm" "")
OPTIMIZERS=("adam" "muon")
WEIGHT_DECAYS=(0.00001 0.0001 0.001 0.01 0.1 1.0)

# Calculate indices from array task ID
# 16 base configs = 2 (modulus) × 2 (attention) × 2 (layernorm) × 2 (optimizer)
# Total = 16 × 6 = 96 jobs

BASE_CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / 6))
WD_IDX=$((SLURM_ARRAY_TASK_ID % 6))

# Decode base config index
# Order: modulus → attention → layernorm → optimizer
# idx = m*8 + a*4 + l*2 + o
MODULUS_IDX=$((BASE_CONFIG_IDX / 8))
REMAINDER=$((BASE_CONFIG_IDX % 8))
ATTENTION_IDX=$((REMAINDER / 4))
REMAINDER=$((REMAINDER % 4))
LAYERNORM_IDX=$((REMAINDER / 2))
OPTIMIZER_IDX=$((REMAINDER % 2))

# Get configuration values
MODULUS=${MODULI[$MODULUS_IDX]}
ATTENTION_TYPE=${ATTENTION_TYPES[$ATTENTION_IDX]}
LAYERNORM_FLAG=${LAYERNORM_FLAGS[$LAYERNORM_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPTIMIZER_IDX]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_IDX]}

# Determine LayerNorm string for experiment name
if [ -n "$LAYERNORM_FLAG" ]; then
    LN_STR="ln"
else
    LN_STR="noln"
fi

echo "============================================================================"
echo "ICML 16_Runs Experiment"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Base Config Index: $BASE_CONFIG_IDX"
echo "Weight Decay Index: $WD_IDX"
echo ""
echo "Configuration:"
echo "  Modulus: $MODULUS"
echo "  Attention: $ATTENTION_TYPE"
echo "  LayerNorm: $LN_STR"
echo "  Optimizer: $OPTIMIZER"
echo "  Weight Decay: $WEIGHT_DECAY"
echo ""
echo "Experiment: p${MODULUS}_${ATTENTION_TYPE}_${LN_STR}_${OPTIMIZER}/wd${WEIGHT_DECAY}_seed42"
echo "============================================================================"

# Change to the training scripts directory
cd "$(dirname "$0")/.."

# Run training
$CONDA_ENV/bin/python training_scripts/train_icml_16runs.py \
    --modulus $MODULUS \
    --attention_type $ATTENTION_TYPE \
    $LAYERNORM_FLAG \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --lr 0.001 \
    --n_epochs 50000 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 1 \
    --d_mlp 512 \
    --train_fraction 0.5 \
    --agop_freq 100 \
    --agop_top_k 20 \
    --ntk_subsample 200 \
    --log_freq 100 \
    --device cuda \
    --seed 42 \
    --save_dir ./results

EXIT_CODE=$?

echo ""
echo "============================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully"
else
    echo "Job FAILED with exit code: $EXIT_CODE"
fi
echo "============================================================================"

exit $EXIT_CODE

