#!/bin/bash
#SBATCH --job-name=gop_slingshot
#SBATCH --output=logs/gop_slingshot_%j.out
#SBATCH --error=logs/gop_slingshot_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

# GOP Analysis for Slingshot Mechanism
mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

cd $SLURM_SUBMIT_DIR

python wrapped_train.py --config ../../configs/07_thilak_slingshot.yaml

echo "GOP analysis complete!"

