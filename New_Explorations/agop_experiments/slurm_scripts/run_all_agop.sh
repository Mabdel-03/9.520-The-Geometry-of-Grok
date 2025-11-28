#!/bin/bash
# Submit all AGOP tracking experiments

echo "Submitting AGOP tracking experiments..."
echo "======================================"

# Submit Nanda experiments
echo "Submitting Nanda (modular addition, ReLU transformer)..."
NANDA_JOB=$(sbatch run_nanda_agop.sh | awk '{print $4}')
echo "  Job ID: $NANDA_JOB"

# Submit Softmax experiments
echo "Submitting Softmax (modular addition, standard transformer)..."
SOFTMAX_JOB=$(sbatch run_softmax_agop.sh | awk '{print $4}')
echo "  Job ID: $SOFTMAX_JOB"

# Submit MNIST experiments
echo "Submitting MNIST (image classification, MLP)..."
MNIST_JOB=$(sbatch run_mnist_agop.sh | awk '{print $4}')
echo "  Job ID: $MNIST_JOB"

# Submit Composition experiments (run last)
echo "Submitting Composition (compositional reasoning)..."
COMP_JOB=$(sbatch run_composition_agop.sh | awk '{print $4}')
echo "  Job ID: $COMP_JOB"

echo ""
echo "All experiments submitted!"
echo "======================================"
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: logs/"

