#!/bin/bash
#SBATCH --job-name=test_agop
#SBATCH --output=test_results/test_agop_%j.out
#SBATCH --error=test_results/test_agop_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

# AGOP Experiments - Comprehensive Test Suite
# Tests all 4 datasets (Nanda, Softmax, MNIST, Composition) with all 3 optimizers (AdamW, Muon, SGD)
# Total: 12 quick tests with 200 epochs each

echo "=========================================="
echo "AGOP Experiments - Comprehensive Test Suite"
echo "Testing: 4 datasets × 3 optimizers = 12 tests"
echo "Epochs per test: 200 (quick verification)"
echo "=========================================="

# Activate conda environment (same as existing experiments)
CONDA_ENV=/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

echo "Using conda environment: $CONDA_ENV"
echo "Python: $($CONDA_ENV/bin/python --version)"
echo ""

# Navigate to AGOP experiments directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments

# Create test results directory
mkdir -p test_results

# Run comprehensive test suite
echo "Starting test suite..."
echo ""
$CONDA_ENV/bin/python tests/test_all_experiments.py --all

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All tests PASSED!"
else
    echo "✗ Some tests FAILED (exit code: $EXIT_CODE)"
fi
echo "Check detailed results in: test_results/"
echo "=========================================="

exit $EXIT_CODE

