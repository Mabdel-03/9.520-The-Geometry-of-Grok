#!/bin/bash
#SBATCH --job-name=paper05_qm9
#SBATCH --output=../results/logs/qm9_%j.out
#SBATCH --error=../results/logs/qm9_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

# Paper 05: Omnigrok - QM9 Molecular Properties

mkdir -p ../results/logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Install torch_geometric if needed
pip install torch-geometric torch-scatter torch-sparse -q

# Navigate to qm9 grokking directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/05_liu_et_al_2022_omnigrok/qm9/grokking || exit 1

echo "=========================================="
echo "Paper 05: Omnigrok - QM9 Molecules"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "Training 1000 QM9 molecular graphs"
echo "Architecture: 2-layer GCNN with ReLU"
echo "Optimizer: Adam (corrected)"
echo ""

# Run the QM9 script
python qm9_grokking.py

echo ""
echo "QM9 experiment complete!"

