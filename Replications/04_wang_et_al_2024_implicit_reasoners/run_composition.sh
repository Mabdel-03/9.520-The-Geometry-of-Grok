#!/bin/bash
#SBATCH --job-name=grok_wang_comp
#SBATCH --output=logs/composition_%j.out
#SBATCH --error=logs/composition_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Grokked Transformers are Implicit Reasoners
# Wang et al. (2024)

mkdir -p logs

# Load modules
module load python/3.9
module load cuda/11.8

cd $SLURM_SUBMIT_DIR

# Install dependencies
pip install -r requirements.txt --quiet 2>/dev/null || pip install torch transformers datasets --quiet

# Run composition reasoning task
# This runs the main grokking experiments on knowledge graph reasoning

python src/train.py \
    --task=composition \
    --model_size=base \
    --num_entities=2000 \
    --batch_size=512 \
    --lr=1e-4 \
    --weight_decay=0.1 \
    --max_steps=2000000 \
    --save_steps=10000

echo "Composition task training complete!"

