#!/bin/bash
#
# Master script to run all 10 grokking paper replications
# Submits jobs with dependencies to avoid resource conflicts
#

set -e

echo "=========================================="
echo "Running All Grokking Paper Replications"
echo "=========================================="
echo ""

# Store job IDs
declare -A JOB_IDS

# Paper 01: Power et al. (2022) - OpenAI Grok
echo "[1/10] Submitting Power et al. (2022) - OpenAI Grok..."
cd 01_power_et_al_2022_openai_grok
JOB_IDS[01]=$(sbatch --parsable run_modular_addition.sh)
echo "  Job ID: ${JOB_IDS[01]}"
cd ..

# Paper 02: Liu et al. (2022) - Effective Theory
echo "[2/10] Submitting Liu et al. (2022) - Effective Theory..."
cd 02_liu_et_al_2022_effective_theory
JOB_IDS[02]=$(sbatch --parsable run_toy_model.sh)
echo "  Job ID: ${JOB_IDS[02]}"
cd ..

# Paper 03: Nanda et al. (2023) - Progress Measures
echo "[3/10] Submitting Nanda et al. (2023) - Progress Measures..."
cd 03_nanda_et_al_2023_progress_measures
JOB_IDS[03]=$(sbatch --parsable run_modular_addition.sh)
echo "  Job ID: ${JOB_IDS[03]}"
cd ..

# Paper 04: Wang et al. (2024) - Implicit Reasoners
echo "[4/10] Submitting Wang et al. (2024) - Knowledge Graphs..."
cd 04_wang_et_al_2024_implicit_reasoners
JOB_IDS[04]=$(sbatch --parsable run_composition.sh)
echo "  Job ID: ${JOB_IDS[04]}"
cd ..

# Paper 05: Liu et al. (2022) - Omnigrok
echo "[5/10] Submitting Liu et al. (2022) - Omnigrok..."
cd 05_liu_et_al_2022_omnigrok
JOB_IDS[05]=$(sbatch --parsable run_mnist.sh)
echo "  Job ID: ${JOB_IDS[05]}"
cd ..

# Paper 06: Humayun et al. (2024) - Deep Networks
echo "[6/10] Submitting Humayun et al. (2024) - Deep Networks..."
cd 06_humayun_et_al_2024_deep_networks
JOB_IDS[06]=$(sbatch --parsable run_mnist_mlp.sh)
echo "  Job ID: ${JOB_IDS[06]}"
cd ..

# Paper 07: Thilak et al. (2022) - Slingshot
echo "[7/10] Submitting Thilak et al. (2022) - Slingshot..."
cd 07_thilak_et_al_2022_slingshot
JOB_IDS[07]=$(sbatch --parsable run_slingshot.sh)
echo "  Job ID: ${JOB_IDS[07]}"
cd ..

# Paper 08: Doshi et al. (2024) - Modular Polynomials
echo "[8/10] Submitting Doshi et al. (2024) - Modular Polynomials..."
cd 08_doshi_et_al_2024_modular_polynomials
JOB_IDS[08]=$(sbatch --parsable run_addition.sh)
echo "  Job ID: ${JOB_IDS[08]}"
cd ..

# Paper 09: Levi et al. (2023) - Linear Estimators
echo "[9/10] Submitting Levi et al. (2023) - Linear Estimators..."
cd 09_levi_et_al_2023_linear_estimators
JOB_IDS[09]=$(sbatch --parsable run_linear_1layer.sh)
echo "  Job ID: ${JOB_IDS[09]}"
cd ..

# Paper 10: Minegishi et al. (2023) - Lottery Tickets
echo "[10/10] Submitting Minegishi et al. (2023) - Lottery Tickets..."
cd 10_minegishi_et_al_2023_grokking_tickets
JOB_IDS[10]=$(sbatch --parsable run_lottery_ticket.sh)
echo "  Job ID: ${JOB_IDS[10]}"
cd ..

echo ""
echo "=========================================="
echo "All Jobs Submitted Successfully!"
echo "=========================================="
echo ""
echo "Job IDs:"
for i in {01..10}; do
    echo "  Paper $i: ${JOB_IDS[$i]}"
done
echo ""
echo "Monitor progress with: squeue -u $USER"
echo "Check specific job: squeue -j JOB_ID"
echo ""
echo "To generate plots after completion, run:"
echo "  python analyze_all_replications.py"
echo ""

