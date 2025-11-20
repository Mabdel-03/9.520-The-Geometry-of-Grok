#!/bin/bash
#SBATCH --job-name=paper05_mnist
#SBATCH --output=../results/logs/mnist_corrected_%j.out
#SBATCH --error=../results/logs/mnist_corrected_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Paper 05: Omnigrok - MNIST with Adam optimizer (corrected)

mkdir -p ../results/logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to mnist grokking directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/05_liu_et_al_2022_omnigrok/mnist/grokking || exit 1

echo "==========================================
"
echo "Paper 05: Omnigrok - MNIST (Corrected with Adam)"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "Training 1000 MNIST samples with Adam optimizer"
echo "Expected: Grokking around 50K-100K steps"
echo ""

# Run the corrected MNIST script
python mnist_grokking_logged.py

echo ""
echo "MNIST experiment complete!"
echo "Results saved to: ../../results/logs/training_history.json"

