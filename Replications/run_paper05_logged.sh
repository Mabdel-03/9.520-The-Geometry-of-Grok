#!/bin/bash
#SBATCH --job-name=grok_paper05_mnist
#SBATCH --partition=use-everything
#SBATCH --output=05_liu_et_al_2022_omnigrok/logs/mnist_logged_%j.out
#SBATCH --error=05_liu_et_al_2022_omnigrok/logs/mnist_logged_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

echo "=========================================="
echo "Paper 05: Liu et al. (2022) - Omnigrok MNIST"
echo "With proper logging to training_history.json"
echo "=========================================="

# Create logs directory
mkdir -p 05_liu_et_al_2022_omnigrok/logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/05_liu_et_al_2022_omnigrok/mnist/grokking

echo "Working directory: $(pwd)"
echo "Running MNIST grokking experiment with 1000 training samples..."
echo "Expected to show grokking after ~50K-100K steps"

# Run the logged version
python mnist_grokking_logged.py

echo "Training complete! Check ../../logs/training_history.json"
ls -lh ../../logs/training_history.json

