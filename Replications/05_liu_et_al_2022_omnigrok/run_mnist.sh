#!/bin/bash
#SBATCH --job-name=grok_omnigrok_mnist
#SBATCH --output=logs/mnist_%j.out
#SBATCH --error=logs/mnist_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Omnigrok: Grokking Beyond Algorithmic Data
# Liu et al. (2022)

mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

cd $SLURM_SUBMIT_DIR

# Run MNIST grokking experiment
# Uses reduced training set (1k samples) to induce grokking

cd mnist/grokking
jupyter nbconvert --to script mnist-grokking.ipynb
python mnist-grokking.py

echo "MNIST grokking experiment complete!"

