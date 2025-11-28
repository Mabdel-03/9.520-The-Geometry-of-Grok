#!/bin/bash
#SBATCH --job-name=test_onehot
#SBATCH --output=test_results/test_onehot_%j.out
#SBATCH --error=test_results/test_onehot_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2

# Quick training test with one-hot AGOP (CPU, no CUDA issues)

echo "=========================================="
echo "Quick Training Test: One-Hot AGOP"
echo "=========================================="

# Activate conda environment
CONDA_ENV=/om/scratch/Tue/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

echo "Environment: $CONDA_ENV"
echo "Python: $($CONDA_ENV/bin/python --version)"
echo ""

cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments

# Create test_results in the agop_experiments directory
mkdir -p /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments/test_results

# Test 1: Nanda MLP (100 epochs)
echo "Test 1/4: Nanda MLP..."
$CONDA_ENV/bin/python training_scripts/train_nanda_agop.py \
    --architecture mlp \
    --optimizer adamw \
    --p 97 \
    --n_epochs 100 \
    --agop_freq 25 \
    --log_freq 10 \
    --device cpu \
    --save_dir ./test_results \
    --experiment_name test_nanda_mlp

# Test 2: Nanda Transformer (100 epochs)
echo "Test 2/4: Nanda Transformer..."
$CONDA_ENV/bin/python training_scripts/train_nanda_agop.py \
    --architecture transformer \
    --optimizer adamw \
    --p 97 \
    --n_epochs 100 \
    --agop_freq 25 \
    --log_freq 10 \
    --device cpu \
    --save_dir ./test_results \
    --experiment_name test_nanda_transformer

# Test 3: MNIST (100 epochs)
echo "Test 3/4: MNIST MLP..."
$CONDA_ENV/bin/python training_scripts/train_mnist_agop.py \
    --optimizer adamw \
    --train_points 500 \
    --n_epochs 100 \
    --agop_freq 25 \
    --agop_subsample 250 \
    --log_freq 10 \
    --device cpu \
    --save_dir ./test_results \
    --experiment_name test_mnist

# Test 4: Composition (100 epochs)
echo "Test 4/4: Composition MLP..."
$CONDA_ENV/bin/python training_scripts/train_composition_agop.py \
    --optimizer adamw \
    --n_facts 200 \
    --n_epochs 100 \
    --agop_freq 25 \
    --log_freq 10 \
    --device cpu \
    --save_dir ./test_results \
    --experiment_name test_composition

echo ""
echo "=========================================="
echo "All 4 training tests completed!"
echo "Check results in: test_results/"
echo "=========================================="

