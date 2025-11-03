#!/bin/bash
#SBATCH --job-name=grok_humayun_mnist
#SBATCH --output=logs/mnist_mlp_%j.out
#SBATCH --error=logs/mnist_mlp_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Deep Networks Always Grok and Here is Why
# Humayun et al. (2024)

mkdir -p logs checkpoints data

# Load CUDA module
module load cuda/11.8

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

cd $SLURM_SUBMIT_DIR

# Run MNIST with MLP
# Reduced training set to observe grokking
python train.py \
    --model=mlp \
    --dataset=mnist \
    --train_size=1000 \
    --batch_size=200 \
    --lr=0.001 \
    --weight_decay=0.01 \
    --n_epochs=100000 \
    --log_interval=100 \
    --device=cuda \
    --seed=42

echo "MNIST MLP training complete!"

