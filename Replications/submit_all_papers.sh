#!/bin/bash
#
# Submit all paper replications from their respective directories
# This ensures SLURM_SUBMIT_DIR is set correctly for each job
#

set -e

BASE_DIR="$(pwd)"
echo "=========================================="
echo "Submitting All Grokking Paper Replications"
echo "Base directory: $BASE_DIR"
echo "=========================================="
echo ""

# Store job IDs
declare -A JOB_IDS

# Paper 01: Power et al. (2022)
echo "[1/10] Paper 01 - Power et al. (2022)..."
cd "$BASE_DIR/01_power_et_al_2022_openai_grok"
JOB_IDS[01]=$(sbatch --parsable run_modular_addition.sh)
echo "  Submitted from: $(pwd)"
echo "  Job ID: ${JOB_IDS[01]}"

# Paper 02: Liu et al. (2022) - Effective Theory  
echo "[2/10] Paper 02 - Liu et al. (2022)..."
cd "$BASE_DIR/02_liu_et_al_2022_effective_theory"
JOB_IDS[02]=$(sbatch --parsable run_toy_model.sh)
echo "  Submitted from: $(pwd)"
echo "  Job ID: ${JOB_IDS[02]}"

# Paper 03: Nanda et al. (2023)
echo "[3/10] Paper 03 - Nanda et al. (2023)..."
cd "$BASE_DIR/03_nanda_et_al_2023_progress_measures"
JOB_IDS[03]=$(sbatch --parsable run_modular_addition.sh)
echo "  Submitted from: $(pwd)"
echo "  Job ID: ${JOB_IDS[03]}"

# Paper 04: Wang et al. (2024)
echo "[4/10] Paper 04 - Wang et al. (2024)..."
cd "$BASE_DIR/04_wang_et_al_2024_implicit_reasoners"
JOB_IDS[04]=$(sbatch --parsable run_composition.sh)
echo "  Submitted from: $(pwd)"
echo "  Job ID: ${JOB_IDS[04]}"

# Paper 05: Liu et al. (2022) - Omnigrok
echo "[5/10] Paper 05 - Liu et al. (2022) Omnigrok..."
cd "$BASE_DIR/05_liu_et_al_2022_omnigrok"
JOB_IDS[05]=$(sbatch --parsable run_mnist.sh)
echo "  Submitted from: $(pwd)"
echo "  Job ID: ${JOB_IDS[05]}"

# Paper 06: Humayun et al. (2024)
echo "[6/10] Paper 06 - Humayun et al. (2024)..."
cd "$BASE_DIR/06_humayun_et_al_2024_deep_networks"
JOB_IDS[06]=$(sbatch --parsable run_mnist_mlp.sh)
echo "  Submitted from: $(pwd)"
echo "  Job ID: ${JOB_IDS[06]}"

# Paper 07: Thilak et al. (2022)
echo "[7/10] Paper 07 - Thilak et al. (2022)..."
cd "$BASE_DIR/07_thilak_et_al_2022_slingshot"
JOB_IDS[07]=$(sbatch --parsable run_slingshot.sh)
echo "  Submitted from: $(pwd)"
echo "  Job ID: ${JOB_IDS[07]}"

# Paper 08: Doshi et al. (2024)
echo "[8/10] Paper 08 - Doshi et al. (2024)..."
cd "$BASE_DIR/08_doshi_et_al_2024_modular_polynomials"
JOB_IDS[08]=$(sbatch --parsable run_addition.sh)
echo "  Submitted from: $(pwd)"
echo "  Job ID: ${JOB_IDS[08]}"

# Paper 09: Levi et al. (2023)
echo "[9/10] Paper 09 - Levi et al. (2023)..."
cd "$BASE_DIR/09_levi_et_al_2023_linear_estimators"
JOB_IDS[09]=$(sbatch --parsable run_linear_1layer.sh)
echo "  Submitted from: $(pwd)"
echo "  Job ID: ${JOB_IDS[09]}"

# Paper 10: Minegishi et al. (2023)
echo "[10/10] Paper 10 - Minegishi et al. (2023)..."
cd "$BASE_DIR/10_minegishi_et_al_2023_grokking_tickets"
JOB_IDS[10]=$(sbatch --parsable run_lottery_ticket.sh)
echo "  Submitted from: $(pwd)"
echo "  Job ID: ${JOB_IDS[10]}"

cd "$BASE_DIR"

echo ""
echo "=========================================="
echo "All Jobs Submitted!"
echo "=========================================="
echo ""
echo "Job IDs:"
for i in {01..10}; do
    echo "  Paper $i: ${JOB_IDS[$i]}"
done
echo ""
echo "Monitor: squeue -u $USER"
echo ""

