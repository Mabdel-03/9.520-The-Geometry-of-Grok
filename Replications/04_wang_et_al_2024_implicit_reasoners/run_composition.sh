#!/bin/bash
#SBATCH --job-name=grok_wang_comp
#SBATCH --partition=use-everything
#SBATCH --output=composition_%j.out
#SBATCH --error=composition_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

# Grokked Transformers are Implicit Reasoners
# Wang et al. (2024)

mkdir -p logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Ensure we're in the right directory
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR="$(pwd)"
fi
cd "$SLURM_SUBMIT_DIR"

echo "Working directory: $(pwd)"
ls main.py || { echo "ERROR: main.py not found!"; exit 1; }

# Install simpletransformers (this repo includes custom implementation)
pip install simpletransformers pandas --quiet 2>/dev/null || echo "Dependencies check"

# Run main.py which contains the training logic
# This trains on knowledge graph composition task
python main.py

echo "Composition task training complete!"

