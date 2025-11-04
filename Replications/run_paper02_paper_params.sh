#!/bin/bash
#SBATCH --job-name=grok_paper02_proper
#SBATCH --partition=use-everything
#SBATCH --output=02_liu_et_al_2022_effective_theory/logs/paper02_proper_%j.out
#SBATCH --error=02_liu_et_al_2022_effective_theory/logs/paper02_proper_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2

echo "=========================================="
echo "Paper 02: Liu et al. (2022) - Effective Theory"
echo "Using paper's exact parameters (train_add)"
echo "=========================================="

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/02_liu_et_al_2022_effective_theory

# Run with paper's exact parameters (from Figure 4)
python run_paper02_proper.py \
    --p=10 \
    --reprs_dim=1 \
    --train_num=45 \
    --steps=5000 \
    --eta_reprs=1e-3 \
    --eta_dec=1e-4 \
    --weight_decay_reprs=0.0 \
    --weight_decay_dec=0.0 \
    --seed=58 \
    --device=cuda

echo "Done!"

