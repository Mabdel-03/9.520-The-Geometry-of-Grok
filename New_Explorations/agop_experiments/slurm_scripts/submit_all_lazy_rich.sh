#!/bin/bash
#
# Submit all experiments with AGOP + Lazy-Rich tracking
# 
# This script submits all 60 experiments:
#   - MNIST: 12 experiments (3 optimizers × 4 weight decays)
#   - Nanda: 24 experiments (2 archs × 3 optimizers × 4 weight decays)
#   - Softmax: 24 experiments (2 archs × 3 optimizers × 4 weight decays)
#
# Each experiment will compute:
#   - Input-gradient AGOP metrics
#   - NTK distance from initialization (lazy→rich transition)
#   - Weight norm evolution
#   - Feature kernel distance
#
# Reference: Kumar et al. (2024) "Grokking as the Transition from Lazy to Rich"
# https://arxiv.org/abs/2310.06110
#

set -e

echo "=============================================="
echo "Submitting all AGOP + Lazy-Rich experiments"
echo "=============================================="

# Make sure logs directory exists
mkdir -p logs

# Submit MNIST experiments (12 jobs)
echo ""
echo "Submitting MNIST experiments (12 jobs)..."
MNIST_JOB=$(sbatch run_mnist_full_sweep.sh | awk '{print $4}')
echo "  MNIST job array submitted: $MNIST_JOB"

# Submit Nanda experiments (24 jobs)
echo ""
echo "Submitting Nanda experiments (24 jobs)..."
NANDA_JOB=$(sbatch run_nanda_full_sweep.sh | awk '{print $4}')
echo "  Nanda job array submitted: $NANDA_JOB"

# Submit Softmax experiments (24 jobs)
echo ""
echo "Submitting Softmax experiments (24 jobs)..."
SOFTMAX_JOB=$(sbatch run_softmax_full_sweep.sh | awk '{print $4}')
echo "  Softmax job array submitted: $SOFTMAX_JOB"

echo ""
echo "=============================================="
echo "All experiments submitted!"
echo "=============================================="
echo ""
echo "Total jobs: 60"
echo "  - MNIST:   $MNIST_JOB (12 jobs)"
echo "  - Nanda:   $NANDA_JOB (24 jobs)"
echo "  - Softmax: $SOFTMAX_JOB (24 jobs)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: ./logs/"
echo ""
echo "After completion, run analysis notebooks:"
echo "  - analysis/analyze_mnist_experiments.ipynb"
echo "  - analysis/analyze_nanda_experiments.ipynb"
echo "  - analysis/analyze_softmax_experiments.ipynb"
echo "=============================================="









