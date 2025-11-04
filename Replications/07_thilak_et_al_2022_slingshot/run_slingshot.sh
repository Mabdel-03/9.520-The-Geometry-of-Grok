#!/bin/bash
#SBATCH --job-name=grok_slingshot
#SBATCH --output=slingshot_%j.out
#SBATCH --error=slingshot_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# The Slingshot Mechanism
# Thilak et al. (2022)
# NOTE: Extended to 300K epochs to observe full slingshot/grokking cycle

mkdir -p logs checkpoints

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Ensure we're in the right directory
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR="$(pwd)"
fi
cd "$SLURM_SUBMIT_DIR"

echo "Working directory: $(pwd)"
ls train.py || { echo "ERROR: train.py not found!"; exit 1; }

# Run Slingshot experiment with weight decay to trigger grokking
# The slingshot paper shows grokking requires weight decay
python train.py \
    --p=97 \
    --train_fraction=0.5 \
    --optimizer=adam \
    --lr=0.001 \
    --weight_decay=1.0 \
    --n_epochs=300000 \
    --log_interval=500 \
    --device=cuda \
    --seed=42

echo "Slingshot experiment complete!"
echo "Check logs/training_history.json for last_layer_norm cycles"

