#!/bin/bash
#SBATCH --job-name=gop_humayun
#SBATCH --output=logs/gop_humayun_%j.out
#SBATCH --error=logs/gop_humayun_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

# GOP Analysis for Deep Networks Always Grok (MNIST MLP)
mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

cd $SLURM_SUBMIT_DIR

python wrapped_train.py --config ../../configs/06_humayun_deep.yaml

echo "GOP analysis complete!"

