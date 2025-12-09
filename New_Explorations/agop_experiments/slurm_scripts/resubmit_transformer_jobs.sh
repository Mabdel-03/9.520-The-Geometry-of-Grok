#!/bin/bash
#
# Resubmit failed transformer jobs with increased memory (96GB)
# These jobs OOM'd during NTK computation
#

set -e

echo "=============================================="
echo "Resubmitting failed Transformer jobs (96GB RAM)"
echo "=============================================="

mkdir -p logs

# Cancel any pending softmax jobs first (they'll also need more memory)
echo "Canceling pending softmax jobs..."
scancel 44408376 2>/dev/null || true

# Resubmit Nanda transformer jobs (indices 12-23)
echo ""
echo "Submitting Nanda Transformer jobs (12-23)..."
NANDA_JOB=$(sbatch --array=12-23 run_nanda_full_sweep.sh | awk '{print $4}')
echo "  Nanda transformer jobs: $NANDA_JOB"

# Resubmit all Softmax jobs with new memory allocation
echo ""
echo "Submitting Softmax jobs (0-23) with 96GB..."
SOFTMAX_JOB=$(sbatch run_softmax_full_sweep.sh | awk '{print $4}')
echo "  Softmax jobs: $SOFTMAX_JOB"

echo ""
echo "=============================================="
echo "Resubmission complete!"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "=============================================="









