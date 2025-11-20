#!/bin/bash
#SBATCH --job-name=grok_humayun_cifar10_cnn
#SBATCH --output=cifar10_cnn_%j.out
#SBATCH --error=cifar10_cnn_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

# Deep Networks Always Grok and Here is Why - CIFAR-10 with CNN
# Humayun et al. (2024)

cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/06_humayun_et_al_2024_deep_networks/scripts

mkdir -p ../logs ../checkpoints ../data ../results

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Run CIFAR-10 with CNN and adversarial testing
# Using reduced training set (5000 samples) to observe grokking
python train.py \
    --model=cnn \
    --dataset=cifar10 \
    --train_size=5000 \
    --batch_size=128 \
    --lr=0.001 \
    --weight_decay=0.0 \
    --n_epochs=100000 \
    --log_interval=100 \
    --save_dir=../results/cifar10_cnn/checkpoints \
    --data_dir=../data \
    --device=cuda \
    --seed=42 \
    --enable_adversarial \
    --adv_eval_batches=20

# Move logs to results directory
mv ../logs/training_history.json ../results/cifar10_cnn/

echo "CIFAR-10 CNN with adversarial testing complete!"

