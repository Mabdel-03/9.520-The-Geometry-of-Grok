#!/bin/bash
# Submit all extended weight decay sweep experiments
# Weight decays: 0, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100 (9 values)
#
# MNIST:   1 arch × 3 opts × 9 WDs = 27 jobs
# Nanda:   2 arch × 3 opts × 9 WDs = 54 jobs
# Softmax: 2 arch × 3 opts × 9 WDs = 54 jobs
# Total: 135 jobs

cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments/slurm_scripts

echo "Submitting Extended Weight Decay Sweep (135 total jobs)"
echo "Weight decays: 0, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100"
echo ""

# Submit MNIST (27 jobs)
echo "Submitting MNIST experiments (27 jobs)..."
MNIST_JOB=$(sbatch run_mnist_wd_sweep.sh | awk '{print $4}')
echo "  MNIST Job ID: $MNIST_JOB"

# Submit Nanda (54 jobs)
echo "Submitting Nanda experiments (54 jobs)..."
NANDA_JOB=$(sbatch run_nanda_wd_sweep.sh | awk '{print $4}')
echo "  Nanda Job ID: $NANDA_JOB"

# Submit Softmax (54 jobs)
echo "Submitting Softmax experiments (54 jobs)..."
SOFTMAX_JOB=$(sbatch run_softmax_wd_sweep.sh | awk '{print $4}')
echo "  Softmax Job ID: $SOFTMAX_JOB"

echo ""
echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Results will be saved to:"
echo "  - results/mnist_wd_sweep/"
echo "  - results/nanda_wd_sweep/"
echo "  - results/softmax_wd_sweep/"







