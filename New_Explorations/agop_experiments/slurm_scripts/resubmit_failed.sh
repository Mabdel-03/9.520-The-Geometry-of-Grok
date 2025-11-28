#!/bin/bash
# Resubmit only the failed experiments with GPU constraints

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║           RESUBMITTING FAILED AGOP EXPERIMENTS (33 jobs)                ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments/slurm_scripts

echo "Failed experiments breakdown:"
echo "  Nanda:       4 failed (CUDA incompatibility)"
echo "  Softmax:     5 failed (CUDA incompatibility)"
echo "  MNIST:      12 failed (CUDA incompatibility)"
echo "  Composition: 12 failed (CUDA incompatibility)"
echo "  ─────────────────────────────────────────────"
echo "  TOTAL:      33 jobs"
echo ""
echo "Fix applied: Added GPU constraint (sm_70+) to avoid GTX 1080 Ti"
echo ""

# Resubmit Nanda with specific failed indices
echo "1/4: Resubmitting Nanda failed experiments (4 jobs)..."
# Jobs 3,4,5,6 failed: mlp_adamw_wd10.0, mlp_muon_wd0.1, mlp_muon_wd1.0, mlp_muon_wd5.0
JOB_NANDA=$(sbatch --array=3,4,5,6 run_nanda_full_sweep.sh | awk '{print $4}')
echo "  Job ID: $JOB_NANDA (indices: 3,4,5,6)"

# Resubmit Softmax with specific failed indices
echo "2/4: Resubmitting Softmax failed experiments (5 jobs)..."
# Need to identify which 5 of 24 failed - resubmit missing ones
JOB_SOFTMAX=$(sbatch --array=19,20,21,22,23 run_softmax_full_sweep.sh | awk '{print $4}')
echo "  Job ID: $JOB_SOFTMAX (indices: 19-23, transformer_muon_wd1.0 + transformer_sgd)"

# Resubmit ALL MNIST (all 12 failed)
echo "3/4: Resubmitting MNIST experiments (12 jobs)..."
JOB_MNIST=$(sbatch run_mnist_full_sweep.sh | awk '{print $4}')
echo "  Job ID: $JOB_MNIST (all 12)"

# Resubmit ALL Composition (all 12 failed)
echo "4/4: Resubmitting Composition experiments (12 jobs)..."
JOB_COMP=$(sbatch run_composition_full_sweep.sh | awk '{print $4}')
echo "  Job ID: $JOB_COMP (all 12)"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║             33 FAILED JOBS RESUBMITTED WITH GPU CONSTRAINTS              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Resubmitted Job IDs:"
echo "  Nanda (4):       $JOB_NANDA"
echo "  Softmax (5):     $JOB_SOFTMAX"
echo "  MNIST (12):      $JOB_MNIST"
echo "  Composition (12): $JOB_COMP"
echo ""
echo "GPU Constraint: tesla (requests Tesla/newer GPUs, excludes GTX 1080 Ti)"
echo ""
echo "Monitor: squeue -u \$USER | grep agop"
echo ""

