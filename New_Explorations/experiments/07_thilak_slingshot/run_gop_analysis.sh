#!/bin/bash
#SBATCH --job-name=gop_slingshot
#SBATCH --output=logs/gop_slingshot_%j.out
#SBATCH --error=logs/gop_slingshot_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# GOP Analysis for Slingshot Mechanism
mkdir -p logs
module load python/3.9 cuda/11.8
cd $SLURM_SUBMIT_DIR

python wrapped_train.py --config ../../configs/07_thilak_slingshot.yaml

echo "GOP analysis complete!"

