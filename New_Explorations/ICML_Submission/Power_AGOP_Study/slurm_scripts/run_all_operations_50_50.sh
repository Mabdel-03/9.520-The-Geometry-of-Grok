#!/bin/bash
#SBATCH --job-name=agop_all_50
#SBATCH --output=/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/ICML_Submission/Power_AGOP_Study/slurm_scripts/logs/agop_all_50_%A_%a.out
#SBATCH --error=/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/ICML_Submission/Power_AGOP_Study/slurm_scripts/logs/agop_all_50_%A_%a.err
#SBATCH --array=0-431
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GEFORCERTX2080:1
#SBATCH --partition=normal

# ============================================================================
# All Operations AGOP Study - 50/50 Train/Test Split
# ============================================================================
#
# Runs all 9 modular arithmetic operations with 50/50 train/test split:
#   - add: x + y
#   - sub: x - y
#   - mul: x * y
#   - div: x / y
#   - cubic: x³ + xy
#   - quadratic: a² + b
#   - symmetric_cubic: a³ + b³
#   - mixed_poly: a² + ab + b²
#   - pure_cubic: x³
#
# Experimental factors (432 total experiments):
#   - Operations: 9
#   - Architecture: transformer, mlp (2)
#   - Optimizer: adamw, muon (2)
#   - Input type: discrete, onehot (2)
#   - Weight decay: 0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 (6)
#
# Array task mapping:
#   task_id = op_idx * 48 + arch_idx * 24 + opt_idx * 12 + input_idx * 6 + wd_idx
#
# ============================================================================

# Activate conda environment
CONDA_ENV=/om/scratch/Mon/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

# Base directory
BASE_DIR=/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/ICML_Submission/Power_AGOP_Study

# Create logs directory if needed
mkdir -p ${BASE_DIR}/slurm_scripts/logs

# Define configuration arrays
OPERATIONS=("add" "sub" "mul" "div" "cubic" "quadratic" "symmetric_cubic" "mixed_poly" "pure_cubic")
ARCHITECTURES=("transformer" "mlp")
OPTIMIZERS=("adamw" "muon")
INPUT_TYPES=("discrete" "onehot")
WEIGHT_DECAYS=(0 0.0001 0.001 0.01 0.1 1.0)

# Train fraction for this script
TRAIN_FRACTION=0.5

# Calculate indices from array task ID
# 432 total = 9 (ops) × 2 (arch) × 2 (opt) × 2 (input) × 6 (wd)
OP_IDX=$((SLURM_ARRAY_TASK_ID / 48))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 48))
ARCH_IDX=$((REMAINDER / 24))
REMAINDER=$((REMAINDER % 24))
OPT_IDX=$((REMAINDER / 12))
REMAINDER=$((REMAINDER % 12))
INPUT_IDX=$((REMAINDER / 6))
WD_IDX=$((REMAINDER % 6))

# Get configuration values
OPERATION=${OPERATIONS[$OP_IDX]}
ARCHITECTURE=${ARCHITECTURES[$ARCH_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
INPUT_TYPE=${INPUT_TYPES[$INPUT_IDX]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_IDX]}

# Result directory includes train fraction
RESULT_DIR="${BASE_DIR}/results_${OPERATION}_50"

echo "============================================================================"
echo "All Operations AGOP Study - 50/50 Split"
echo "============================================================================"
echo "Operation: $OPERATION"
echo "Train Fraction: $TRAIN_FRACTION"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo ""
echo "Configuration:"
echo "  Architecture: $ARCHITECTURE"
echo "  Optimizer: $OPTIMIZER"
echo "  Input Type: $INPUT_TYPE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo ""
echo "Experiment: ${ARCHITECTURE}_${INPUT_TYPE}_${OPTIMIZER}/wd${WEIGHT_DECAY}_seed42"
echo "Results: $RESULT_DIR"
echo "============================================================================"

# Change to the base directory
cd ${BASE_DIR}

# Run training
$CONDA_ENV/bin/python ${BASE_DIR}/training_scripts/train_power_agop.py \
    --modulus 97 \
    --operation $OPERATION \
    --architecture $ARCHITECTURE \
    --input_type $INPUT_TYPE \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --lr 0.001 \
    --n_epochs 50000 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --d_mlp 512 \
    --train_fraction $TRAIN_FRACTION \
    --agop_freq 100 \
    --agop_top_k 20 \
    --log_freq 100 \
    --device cuda \
    --seed 42 \
    --save_dir $RESULT_DIR

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
