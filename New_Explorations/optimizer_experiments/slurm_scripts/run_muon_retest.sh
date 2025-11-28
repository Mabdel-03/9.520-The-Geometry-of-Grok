#!/bin/bash
# Rerun ONLY Muon experiments with official implementation
# Tests if official Muon performs better than custom version

echo "Resubmitting Muon experiments with OFFICIAL implementation..."

# Create logs directory
mkdir -p logs

# Define weight decay values to test
WEIGHT_DECAYS=(0.0 0.01 0.1 0.5 1.0 2.0 5.0)

# Only Muon optimizer (official implementation with lr=0.02)
OPTIMIZER=muonw

# Submit jobs for all weight decays
for wd in "${WEIGHT_DECAYS[@]}"; do
    echo "Submitting: $OPTIMIZER with weight_decay=$wd (lr=0.02)"
    sbatch run_softmax_single.sh $OPTIMIZER $wd
    sleep 0.5
done

echo "All Muon jobs submitted!"
echo "Total jobs: ${#WEIGHT_DECAYS[@]}"
echo ""
echo "Note: Using official Muon with lr=0.02 (was 0.001)"
echo "Check job status with: squeue -u \$USER | grep softmax"

