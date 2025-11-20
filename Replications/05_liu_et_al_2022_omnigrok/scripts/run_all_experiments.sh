#!/bin/bash

# Master script to run all Paper 05 (Omnigrok) experiments
# This submits 6 experiments as separate SLURM jobs

echo "============================================"
echo "Paper 05: Omnigrok - Running All Experiments"
echo "============================================"
echo ""
echo "This will submit 6 experiments:"
echo "1. MNIST (corrected with Adam)"
echo "2. IMDb Sentiment"
echo "3. QM9 Molecules"
echo "4. Teacher-Student"
echo "5. Modular Addition"
echo "6. MNIST Representation"
echo ""

# Navigate to scripts directory
cd "$(dirname "$0")" || exit 1

# Make all scripts executable
chmod +x run_mnist_corrected.sh
chmod +x run_imdb.sh
chmod +x run_qm9.sh
chmod +x run_teacher_student.sh
chmod +x run_modular_addition.sh
chmod +x run_mnist_repr.sh

# Submit all jobs
echo "Submitting MNIST (corrected)..."
JOB1=$(sbatch run_mnist_corrected.sh | awk '{print $4}')
echo "  Job ID: $JOB1"

echo "Submitting IMDb..."
JOB2=$(sbatch run_imdb.sh | awk '{print $4}')
echo "  Job ID: $JOB2"

echo "Submitting QM9..."
JOB3=$(sbatch run_qm9.sh | awk '{print $4}')
echo "  Job ID: $JOB3"

echo "Submitting Teacher-Student..."
JOB4=$(sbatch run_teacher_student.sh | awk '{print $4}')
echo "  Job ID: $JOB4"

echo "Submitting Modular Addition..."
JOB5=$(sbatch run_modular_addition.sh | awk '{print $4}')
echo "  Job ID: $JOB5"

echo "Submitting MNIST Representation..."
JOB6=$(sbatch run_mnist_repr.sh | awk '{print $4}')
echo "  Job ID: $JOB6"

echo ""
echo "============================================"
echo "All jobs submitted!"
echo "============================================"
echo ""
echo "Job IDs: $JOB1 $JOB2 $JOB3 $JOB4 $JOB5 $JOB6"
echo ""
echo "Monitor jobs with: squeue -u $(whoami)"
echo "Check logs in: ../results/logs/"
echo ""
echo "Expected completion times:"
echo "  - MNIST: ~2-4 hours"
echo "  - IMDb: ~6-8 hours"
echo "  - QM9: ~8-12 hours"
echo "  - Teacher-Student: ~4-6 hours"
echo "  - Modular Addition: ~6-8 hours"
echo "  - MNIST Representation: ~2-4 hours"
echo ""

