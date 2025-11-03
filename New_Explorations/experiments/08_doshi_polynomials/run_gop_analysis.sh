#!/bin/bash
#SBATCH --job-name=gop_polynomials
#SBATCH --output=logs/gop_polynomials_%j.out
#SBATCH --error=logs/gop_polynomials_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

# GOP Analysis for Modular Polynomials
mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

cd $SLURM_SUBMIT_DIR

python wrapped_train.py --config ../../configs/08_doshi_polynomials.yaml

echo "GOP analysis complete!"

