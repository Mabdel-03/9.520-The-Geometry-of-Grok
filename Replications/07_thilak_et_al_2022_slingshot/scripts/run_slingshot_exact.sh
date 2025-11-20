#!/bin/bash
#SBATCH --job-name=grok_slingshot_exact
#SBATCH --output=slingshot_exact_%j.out
#SBATCH --error=slingshot_exact_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# The Slingshot Mechanism - EXACT PAPER REPLICATION
# Thilak et al. (2022)
# Key fix: weight_decay=0.0 (paper's main claim: Slingshot WITHOUT regularization)

# Get the scripts directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Script directory: $SCRIPT_DIR"
mkdir -p ../results/logs ../results/checkpoints logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

echo "=========================================="
echo "Paper 07: Slingshot Mechanism (EXACT REPLICATION)"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "Key change: weight_decay=0.0 (NOT 1.0)"
echo "Testing paper's claim: Slingshot WITHOUT regularization"
echo "=========================================="

# Verify train.py exists
if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found in $(pwd)"
    ls -la
    exit 1
fi

echo "✅ Found train.py"

# Run with EXACT paper specifications
# Critical: weight_decay=0.0 (paper's key claim)
# Using Adam (not AdamW) to match paper's emphasis on "Adam family"
# Same architecture and dataset as before

python train.py \
    --p=97 \
    --train_fraction=0.5 \
    --d_model=128 \
    --n_heads=4 \
    --n_layers=2 \
    --d_mlp=512 \
    --optimizer=adam \
    --lr=0.001 \
    --weight_decay=0.0 \
    --n_epochs=100000 \
    --log_interval=100 \
    --save_dir=./checkpoints \
    --device=cuda \
    --seed=42

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Copy results to results directory for comparison
    echo "Copying results to ../results/ for comparison..."
    cp -v logs/training_history.json ../results/logs/training_history.json
    
    echo ""
    echo "Results saved to:"
    echo "  - logs/training_history.json (local)"
    echo "  - ../results/logs/training_history.json (for comparison)"
    echo ""
    echo "Next steps:"
    echo "  1. Run: python compare_wd_experiments.py"
    echo "  2. Analyze Slingshot mechanism with WD=0.0"
    echo "  3. Compare with WD=1.0 results"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "Check error log for details"
fi
echo "=========================================="

exit $EXIT_CODE

