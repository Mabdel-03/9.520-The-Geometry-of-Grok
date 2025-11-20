#!/bin/bash

# Master resubmission script for Paper 05 (Omnigrok)
# All experiments with AdamW optimizer (what actually works)

echo "============================================"
echo "Paper 05: Omnigrok - Resubmitting ALL Experiments"
echo "With CORRECT Optimizer: AdamW"
echo "============================================"
echo ""
echo "This will submit 5 experiments:"
echo "1. MNIST (AdamW - final version)"
echo "2. Teacher-Student (AdamW, fixed threshold)"
echo "3. QM9 Molecules (AdamW)"
echo "4. Modular Addition (AdamW)"
echo "5. MNIST Representation (AdamW)"
echo ""

# Navigate to scripts directory
cd "$(dirname "$0")" || exit 1

# Submit all jobs
echo "Submitting MNIST (FINAL - AdamW)..."
JOB1=$(sbatch run_mnist_final.sh | awk '{print $4}')
echo "  Job ID: $JOB1"

echo "Submitting Teacher-Student (fixed)..."
JOB2=$(sbatch run_teacher_student.sh | awk '{print $4}')
echo "  Job ID: $JOB2"

echo "Submitting QM9 (fixed)..."
JOB3=$(sbatch run_qm9.sh | awk '{print $4}')
echo "  Job ID: $JOB3"

echo "Submitting Modular Addition (fixed)..."
JOB4=$(sbatch run_modular_addition.sh | awk '{print $4}')
echo "  Job ID: $JOB4"

echo "Submitting MNIST Representation (confirmed working)..."
JOB5=$(sbatch run_mnist_repr.sh | awk '{print $4}')
echo "  Job ID: $JOB5"

echo ""
echo "============================================"
echo "All jobs submitted!"
echo "============================================"
echo ""
echo "Job IDs: $JOB1 $JOB2 $JOB3 $JOB4 $JOB5"
echo ""
echo "Monitor jobs with: squeue -u $(whoami) | grep paper05"
echo "Check logs in: ../results/logs/"
echo ""
echo "Expected completion times:"
echo "  - MNIST: ~2-4 hours (100K steps)"
echo "  - Teacher-Student: ~4-6 hours"
echo "  - QM9: ~8-12 hours"
echo "  - Modular Addition: ~6-8 hours"
echo "  - MNIST Representation: ~2-4 hours"
echo ""
echo "Expected Results:"
echo "  - MNIST: 100% train, ~89% test (smooth grokking)"
echo "  - Teacher-Student: High train, moderate test"
echo "  - QM9: Clear grokking with small training set"
echo "  - Modular Addition: Sharp grokking transitions"
echo "  - MNIST Repr: Landscape analysis complete"
echo ""

