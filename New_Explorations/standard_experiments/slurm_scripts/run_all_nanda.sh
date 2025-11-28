#!/bin/bash
# Master script to submit all Nanda experiments
# Tests all optimizer/weight_decay combinations

echo "Submitting all Nanda et al. experiments..."

# Create logs directory
mkdir -p logs

# Define weight decay values to test
WEIGHT_DECAYS=(0.0 0.01 0.1 0.5 1.0 2.0 5.0 10.0)

# Define optimizers to test
OPTIMIZERS=(muonw adamw sgd)

# Submit jobs for all combinations
for optimizer in "${OPTIMIZERS[@]}"; do
    for wd in "${WEIGHT_DECAYS[@]}"; do
        echo "Submitting: $optimizer with weight_decay=$wd"
        sbatch run_nanda_single.sh $optimizer $wd
        sleep 0.5  # Small delay to avoid overwhelming scheduler
    done
done

echo "All jobs submitted!"
echo "Total jobs: $((${#OPTIMIZERS[@]} * ${#WEIGHT_DECAYS[@]}))"
echo ""
echo "Check job status with: squeue -u \$USER"
echo "Monitor logs in: logs/"

