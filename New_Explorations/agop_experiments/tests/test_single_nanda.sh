#!/bin/bash
#SBATCH --job-name=test_nanda
#SBATCH --output=test_results/test_nanda_%j.out
#SBATCH --error=test_results/test_nanda_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4

# Quick test of Nanda experiment with AGOP tracking (CPU only, no CUDA issues)

echo "=========================================="
echo "Testing Nanda AGOP Experiment (CPU)"
echo "=========================================="

# Activate conda environment
CONDA_ENV=/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

echo "Python: $($CONDA_ENV/bin/python --version)"
echo "PyTorch: $($CONDA_ENV/bin/python -c 'import torch; print(torch.__version__)')"
echo ""

# Navigate to directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments

# Create test results directory
mkdir -p test_results

# Run single Nanda test with AdamW
echo "Running Nanda with AdamW (200 epochs, CPU)..."
$CONDA_ENV/bin/python training_scripts/train_nanda_agop.py \
    --optimizer adamw \
    --lr 0.001 \
    --weight_decay 1.0 \
    --p 97 \
    --train_fraction 0.3 \
    --n_epochs 200 \
    --agop_freq 50 \
    --log_freq 10 \
    --device cpu \
    --seed 42 \
    --save_dir ./test_results \
    --experiment_name test_nanda_adamw_cpu

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test PASSED!"
    echo "Check results in: test_results/test_nanda_adamw_cpu/"
else
    echo "✗ Test FAILED (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE

