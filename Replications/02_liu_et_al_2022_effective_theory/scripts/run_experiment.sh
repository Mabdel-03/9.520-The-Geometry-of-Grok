#!/bin/bash
#SBATCH --job-name=liu2022_grok
#SBATCH --output=results/experiment_%j.out
#SBATCH --error=results/experiment_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2

# ============================================================================
# Liu et al. (2022): Towards Understanding Grokking
# Unified experiment runner for cluster execution
# ============================================================================

echo "=========================================="
echo "Liu et al. (2022) - Effective Theory"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Create results directory
mkdir -p results

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to script directory
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    SLURM_SUBMIT_DIR="$(pwd)"
fi
cd "$SLURM_SUBMIT_DIR"

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=========================================="

# Run Experiment 1: Toy Model (Main Grokking Demonstration)
echo ""
echo "Running Experiment 1: Toy Model..."
echo "=========================================="

python run_experiment.py \
    --experiment toy_model \
    --p 10 \
    --train_num 45 \
    --reprs_dim 1 \
    --steps 5000 \
    --eta_reprs 1e-3 \
    --eta_dec 1e-4 \
    --weight_decay_reprs 0.0 \
    --weight_decay_dec 0.0 \
    --seed 58 \
    --device cuda \
    --output_dir results

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Experiment completed successfully!"
    echo "Results saved to: results/experiment_1_toy_model/"
else
    echo "❌ Experiment failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

