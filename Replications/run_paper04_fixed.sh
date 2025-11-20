#!/bin/bash
#SBATCH --job-name=paper04_fixed
#SBATCH --partition=normal
#SBATCH --output=04_wang_et_al_2024_implicit_reasoners/results/logs/composition_minimal_%j.out
#SBATCH --error=04_wang_et_al_2024_implicit_reasoners/results/logs/composition_minimal_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

echo "=========================================="
echo "Paper 04: Wang et al. (2024) - FIXED VERSION"
echo "Compositional Reasoning on Knowledge Graphs"
echo "Training for 100,000 steps (estimated 6-12 hours)"
echo "=========================================="
echo ""

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/04_wang_et_al_2024_implicit_reasoners

echo "Installing modified libraries..."
# Install modified transformers
cd transformers
pip install -e . --quiet 2>/dev/null || echo "Transformers already installed"
cd ..

# Install modified simpletransformers  
cd simpletransformers
pip install -e . --quiet 2>/dev/null || echo "Simpletransformers already installed"
cd ..

echo "✅ Libraries ready"
echo ""

# Verify data exists
if [ ! -d "data/composition_minimal" ]; then
    echo "❌ ERROR: data/composition_minimal not found!"
    exit 1
fi

echo "✅ Dataset verified: data/composition_minimal"
echo "   - train.json: $(wc -l < data/composition_minimal/train.json) lines"
echo "   - valid.json: $(wc -l < data/composition_minimal/valid.json) lines"
echo "   - test.json: $(wc -l < data/composition_minimal/test.json) lines"
echo ""

# Create output directory
mkdir -p output_dir/composition_minimal
mkdir -p results/logs

echo "Starting training with FIXED configuration..."
echo "Configuration:"
echo "  - Model: GPT-2 (4 layers)"
echo "  - Training steps: 100,000"
echo "  - Batch size: 64 x 8 (gradient accum) = 512 effective"
echo "  - Learning rate: 1e-4"
echo "  - Weight decay: 0.1"
echo "  - Dataset: 181,000 examples"
echo ""

# Run training with FIXED arguments
python scripts/main.py \
    --data_dir=data/composition_minimal \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --init_weights \
    --n_layer=4 \
    --add_tokens \
    --no_dropout \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir \
    --output_dir=output_dir/composition_minimal \
    --train_batch_size=64 \
    --eval_batch_size=64 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-4 \
    --weight_decay=0.1 \
    --max_steps=100000 \
    --save_step=10000 \
    --warmup_steps=1000 \
    --scheduler=linear_schedule_with_warmup \
    --max_seq_length=64 \
    --max_length=64 \
    --manual_seed=42 \
    --fp16

echo ""
echo "=========================================="
echo "Training complete!"
echo "Check output_dir/composition_minimal/ for checkpoints"
echo "=========================================="

