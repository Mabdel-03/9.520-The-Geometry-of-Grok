#!/bin/bash
#SBATCH --job-name=paper05_imdb
#SBATCH --output=../results/logs/imdb_%j.out
#SBATCH --error=../results/logs/imdb_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

# Paper 05: Omnigrok - IMDb Sentiment Analysis

mkdir -p ../results/logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to imdb grokking directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/05_liu_et_al_2022_omnigrok/imdb/grokking || exit 1

echo "=========================================="
echo "Paper 05: Omnigrok - IMDb Sentiment"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "Training 1000 IMDb samples with 2-layer LSTM"
echo "Architecture: embedding_dim=64, hidden_dim=128 (corrected)"
echo "Optimizer: Adam (corrected)"
echo ""

# Check if IMDB dataset exists
if [ ! -f "IMDB Dataset.csv" ]; then
    echo "ERROR: IMDB Dataset.csv not found!"
    echo "Please download it from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    exit 1
fi

# Run the IMDb script
python imdb-grokking

echo ""
echo "IMDb experiment complete!"

