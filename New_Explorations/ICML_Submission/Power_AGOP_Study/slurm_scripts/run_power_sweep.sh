#!/bin/bash
#SBATCH --job-name=power_agop
#SBATCH --output=/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/ICML_Submission/Power_AGOP_Study/slurm_scripts/logs/power_agop_%A_%a.out
#SBATCH --error=/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/ICML_Submission/Power_AGOP_Study/slurm_scripts/logs/power_agop_%A_%a.err
#SBATCH --array=0-47
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GEFORCERTX2080:1
#SBATCH --partition=normal

# ============================================================================
# Power AGOP Study Experiment Sweep
# ============================================================================
#
# Based on Power et al. (2022) "Grokking" experimental setup with AGOP analysis.
#
# Experimental factors (48 total experiments):
#   - Architecture: transformer, mlp (2)
#   - Optimizer: adamw, muon (2)
#   - Input type: discrete, onehot (2)
#   - Weight decay: 0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 (6)
#
# Array task mapping:
#   task_id = arch_idx * 24 + opt_idx * 12 + input_idx * 6 + wd_idx
#
#   arch_idx = task_id / 24
#   opt_idx = (task_id % 24) / 12
#   input_idx = (task_id % 12) / 6
#   wd_idx = task_id % 6
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
ARCHITECTURES=("transformer" "mlp")
OPTIMIZERS=("adamw" "muon")
INPUT_TYPES=("discrete" "onehot")
WEIGHT_DECAYS=(0 0.0001 0.001 0.01 0.1 1.0)

# Calculate indices from array task ID
# 48 total = 2 (arch) × 2 (opt) × 2 (input) × 6 (wd)
ARCH_IDX=$((SLURM_ARRAY_TASK_ID / 24))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 24))
OPT_IDX=$((REMAINDER / 12))
REMAINDER=$((REMAINDER % 12))
INPUT_IDX=$((REMAINDER / 6))
WD_IDX=$((REMAINDER % 6))

# Get configuration values
ARCHITECTURE=${ARCHITECTURES[$ARCH_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
INPUT_TYPE=${INPUT_TYPES[$INPUT_IDX]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_IDX]}

echo "============================================================================"
echo "Power AGOP Study Experiment"
echo "============================================================================"
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
echo "============================================================================"

# Change to the base directory
cd ${BASE_DIR}

# Run training
$CONDA_ENV/bin/python ${BASE_DIR}/training_scripts/train_power_agop.py \
    --modulus 97 \
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
    --train_fraction 0.5 \
    --agop_freq 100 \
    --agop_top_k 20 \
    --log_freq 100 \
    --device cuda \
    --seed 42 \
    --save_dir ${BASE_DIR}/results

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

