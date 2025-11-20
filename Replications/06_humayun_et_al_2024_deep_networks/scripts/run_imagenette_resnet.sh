#!/bin/bash
#SBATCH --job-name=grok_humayun_imagenette
#SBATCH --output=imagenette_resnet_%j.out
#SBATCH --error=imagenette_resnet_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

# Deep Networks Always Grok and Here is Why - Imagenette with ResNet-18
# Humayun et al. (2024)

cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/06_humayun_et_al_2024_deep_networks/scripts

mkdir -p ../logs ../checkpoints ../data ../results

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Run Imagenette with ResNet-18 and adversarial testing
# Using reduced training set to observe grokking
python train.py \
    --model=resnet18 \
    --dataset=imagenette \
    --train_size=5000 \
    --batch_size=64 \
    --lr=0.001 \
    --weight_decay=0.0 \
    --n_epochs=100000 \
    --log_interval=100 \
    --save_dir=../results/imagenette_resnet/checkpoints \
    --data_dir=../data \
    --device=cuda \
    --seed=42 \
    --enable_adversarial \
    --adv_eval_batches=10

# Move logs to results directory
mv ../logs/training_history.json ../results/imagenette_resnet/

echo "Imagenette ResNet-18 with adversarial testing complete!"

