#!/bin/bash
#SBATCH --job-name=grok_omnigrok_mnist
#SBATCH --output=mnist_%j.out
#SBATCH --error=mnist_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Omnigrok: Grokking Beyond Algorithmic Data
# Liu et al. (2022)

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
ls mnist/grokking/mnist-grokking.ipynb || { echo "ERROR: mnist/grokking/mnist-grokking.ipynb not found!"; exit 1; }

# Run MNIST grokking experiment
# Uses reduced training set (1k samples) to induce grokking

cd mnist/grokking
echo "Notebook directory: $(pwd)"

# Convert notebook to Python
jupyter nbconvert --to python mnist-grokking.ipynb

# Remove get_ipython() calls that don't work in scripts
sed -i 's/get_ipython()\.system/#get_ipython().system/g' mnist-grokking.py
sed -i 's/get_ipython()\.run_line_magic/#get_ipython().run_line_magic/g' mnist-grokking.py

# Run the cleaned script
python mnist-grokking.py

echo "MNIST grokking experiment complete!"

