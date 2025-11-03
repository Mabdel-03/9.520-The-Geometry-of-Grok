#!/bin/bash
#SBATCH --job-name=grok_slingshot
#SBATCH --output=logs/slingshot_%j.out
#SBATCH --error=logs/slingshot_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# The Slingshot Mechanism
# Thilak et al. (2022)

mkdir -p logs checkpoints

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

cd $SLURM_SUBMIT_DIR

# Run Slingshot experiment with Adam (no weight decay)
# Track last-layer weight norms for cyclic Slingshot behavior
python train.py \
    --p=97 \
    --train_fraction=0.5 \
    --optimizer=adam \
    --lr=0.001 \
    --weight_decay=0.0 \
    --n_epochs=50000 \
    --log_interval=50 \
    --device=cuda \
    --seed=42

echo "Slingshot experiment complete!"
echo "Check logs/training_history.json for last_layer_norm cycles"

