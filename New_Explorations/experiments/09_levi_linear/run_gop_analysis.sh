#!/bin/bash
#SBATCH --job-name=gop_linear
#SBATCH --output=logs/gop_linear_%j.out
#SBATCH --error=logs/gop_linear_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# GOP Analysis for Linear Estimators (smallest model - good for testing)
mkdir -p logs
module load python/3.9 cuda/11.8
cd $SLURM_SUBMIT_DIR

python wrapped_train.py --config ../../configs/09_levi_linear.yaml

echo "GOP analysis complete!"

