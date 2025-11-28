#!/bin/bash
# Submit ALL AGOP experiments - Complete Matrix
# Total: 72 jobs (24 + 24 + 12 + 12)

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║       SUBMITTING ALL AGOP EXPERIMENTS - FULL MATRIX                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments/slurm_scripts

# Create logs directory
mkdir -p logs

echo "Experiment Matrix:"
echo "  Nanda:       2 architectures × 3 optimizers × 4 weight decays = 24 jobs"
echo "  Softmax:     2 architectures × 3 optimizers × 4 weight decays = 24 jobs"
echo "  MNIST:       1 architecture  × 3 optimizers × 4 weight decays = 12 jobs"
echo "  Composition: 1 architecture  × 3 optimizers × 4 weight decays = 12 jobs"
echo "  ─────────────────────────────────────────────────────────────────────────"
echo "  TOTAL:                                                          72 jobs"
echo ""

# Submit Nanda experiments (24 jobs)
echo "1/4: Submitting Nanda experiments (24 jobs)..."
JOB_NANDA=$(sbatch run_nanda_full_sweep.sh | awk '{print $4}')
echo "  Job ID: $JOB_NANDA"

# Submit Softmax experiments (24 jobs)
echo "2/4: Submitting Softmax experiments (24 jobs)..."
JOB_SOFTMAX=$(sbatch run_softmax_full_sweep.sh | awk '{print $4}')
echo "  Job ID: $JOB_SOFTMAX"

# Submit MNIST experiments (12 jobs)
echo "3/4: Submitting MNIST experiments (12 jobs)..."
JOB_MNIST=$(sbatch run_mnist_full_sweep.sh | awk '{print $4}')
echo "  Job ID: $JOB_MNIST"

# Submit Composition experiments (12 jobs) - Run last
echo "4/4: Submitting Composition experiments (12 jobs)..."
JOB_COMP=$(sbatch run_composition_full_sweep.sh | awk '{print $4}')
echo "  Job ID: $JOB_COMP"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                    ALL 72 JOBS SUBMITTED!                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job IDs:"
echo "  Nanda:       $JOB_NANDA (24 jobs)"
echo "  Softmax:     $JOB_SOFTMAX (24 jobs)"
echo "  MNIST:       $JOB_MNIST (12 jobs)"
echo "  Composition: $JOB_COMP (12 jobs)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'squeue -u \$USER'"
echo ""
echo "Check logs in:"
echo "  tail -f logs/nanda_agop_*.out"
echo ""
echo "Results will be saved to:"
echo "  ../results/nanda/"
echo "  ../results/softmax/"
echo "  ../results/mnist/"
echo "  ../results/composition/"
echo ""


