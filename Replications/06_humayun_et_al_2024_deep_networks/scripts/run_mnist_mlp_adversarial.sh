#!/bin/bash
#SBATCH --job-name=grok_humayun_mnist_adv
#SBATCH --output=mnist_mlp_adv_%j.out
#SBATCH --error=mnist_mlp_adv_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Deep Networks Always Grok and Here is Why - MNIST with Adversarial Testing
# Humayun et al. (2024)

cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/06_humayun_et_al_2024_deep_networks/scripts

mkdir -p ../logs ../checkpoints ../data ../results

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Run MNIST with MLP and adversarial testing
python train.py \
    --model=mlp \
    --dataset=mnist \
    --train_size=1000 \
    --batch_size=200 \
    --lr=0.001 \
    --weight_decay=0.01 \
    --n_epochs=100000 \
    --log_interval=100 \
    --save_dir=../results/mnist_mlp_adv/checkpoints \
    --data_dir=../data \
    --device=cuda \
    --seed=42 \
    --enable_adversarial \
    --adv_eval_batches=20

# Move logs to results directory
mv ../logs/training_history.json ../results/mnist_mlp_adv/

echo "MNIST MLP with adversarial testing complete!"

