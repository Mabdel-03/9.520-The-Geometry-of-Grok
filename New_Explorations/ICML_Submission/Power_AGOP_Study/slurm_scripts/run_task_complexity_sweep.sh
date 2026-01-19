#!/bin/bash
#SBATCH --job-name=task_complexity
#SBATCH --output=/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/ICML_Submission/Power_AGOP_Study/slurm_scripts/logs/task_complexity_%A_%a.out
#SBATCH --error=/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/ICML_Submission/Power_AGOP_Study/slurm_scripts/logs/task_complexity_%A_%a.err
#SBATCH --array=0-15
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GEFORCERTX2080:1
#SBATCH --partition=normal

# ============================================================================
# Task Complexity Spectrum Experiment Sweep (Experiment 4)
# ============================================================================
#
# Tests intermediate-complexity tasks to establish a complexity-grokking curve.
#
# New tasks:
#   - quadratic: f(a,b) = (a² + b) mod 97
#   - symmetric_cubic: f(a,b) = (a³ + b³) mod 97
#   - mixed_poly: f(a,b) = (a² + ab + b²) mod 97
#   - mul: f(a,b) = (a × b) mod 97
#
# Fixed configuration (focused sweep):
#   - Architecture: transformer only
#   - Optimizer: adamw only
#   - Input type: discrete only
#
# Experimental factors (16 total experiments):
#   - Task: mul, quadratic, mixed_poly, symmetric_cubic (4)
#   - Weight decay: 0, 1e-3, 1e-2, 1e-1 (4)
#
# Array task mapping:
#   task_id = operation_idx * 4 + wd_idx
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
OPERATIONS=("mul" "quadratic" "mixed_poly" "symmetric_cubic")
WEIGHT_DECAYS=(0 0.001 0.01 0.1)

# Fixed parameters
ARCHITECTURE="transformer"
OPTIMIZER="adamw"
INPUT_TYPE="discrete"

# Calculate indices from array task ID
# 16 total = 4 (operations) × 4 (weight decays)
OP_IDX=$((SLURM_ARRAY_TASK_ID / 4))
WD_IDX=$((SLURM_ARRAY_TASK_ID % 4))

# Get configuration values
OPERATION=${OPERATIONS[$OP_IDX]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_IDX]}

# Set output directory based on operation
if [ "$OPERATION" == "mul" ]; then
    SAVE_DIR="${BASE_DIR}/results_mul"
elif [ "$OPERATION" == "quadratic" ]; then
    SAVE_DIR="${BASE_DIR}/results_quadratic"
elif [ "$OPERATION" == "mixed_poly" ]; then
    SAVE_DIR="${BASE_DIR}/results_mixed_poly"
elif [ "$OPERATION" == "symmetric_cubic" ]; then
    SAVE_DIR="${BASE_DIR}/results_symmetric_cubic"
fi

echo "============================================================================"
echo "Task Complexity Spectrum Experiment"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo ""
echo "Configuration:"
echo "  Operation: $OPERATION"
echo "  Architecture: $ARCHITECTURE"
echo "  Optimizer: $OPTIMIZER"
echo "  Input Type: $INPUT_TYPE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo ""
echo "Output: ${SAVE_DIR}/${ARCHITECTURE}_${INPUT_TYPE}_${OPTIMIZER}/wd${WEIGHT_DECAY}_seed42"
echo "============================================================================"

# Change to the base directory
cd ${BASE_DIR}

# Run training (shorter epochs for quick turnaround)
$CONDA_ENV/bin/python ${BASE_DIR}/training_scripts/train_power_agop.py \
    --operation $OPERATION \
    --modulus 97 \
    --architecture $ARCHITECTURE \
    --input_type $INPUT_TYPE \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --lr 0.001 \
    --n_epochs 25000 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --d_mlp 512 \
    --train_fraction 0.5 \
    --agop_freq 100 \
    --agop_top_k 20 \
    --log_freq 100 \
    --device cuda \
    --seed 42 \
    --save_dir ${SAVE_DIR}

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
