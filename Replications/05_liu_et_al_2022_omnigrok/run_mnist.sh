#!/bin/bash
#SBATCH --job-name=grok_omnigrok_mnist
#SBATCH --output=logs/mnist_%j.out
#SBATCH --error=logs/mnist_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Omnigrok: Grokking Beyond Algorithmic Data
# Liu et al. (2022)

mkdir -p logs

# Load modules
module load python/3.9
module load cuda/11.8

cd $SLURM_SUBMIT_DIR

# Install dependencies
pip install torch torchvision numpy matplotlib jupyter --quiet

# Run MNIST grokking experiment
# Uses reduced training set (1k samples) to induce grokking

cd mnist/grokking
jupyter nbconvert --to script mnist-grokking.ipynb
python mnist-grokking.py

echo "MNIST grokking experiment complete!"

