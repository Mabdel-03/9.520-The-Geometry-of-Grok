#!/bin/bash
#SBATCH --job-name=grok_paper04_minimal
#SBATCH --partition=use-everything
#SBATCH --output=04_wang_et_al_2024_implicit_reasoners/logs/composition_minimal_%j.out
#SBATCH --error=04_wang_et_al_2024_implicit_reasoners/logs/composition_minimal_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8

echo "=========================================="
echo "Paper 04: Wang et al. (2024) - Implicit Reasoners"
echo "Minimal composition dataset (500 entities)"
echo "=========================================="

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/04_wang_et_al_2024_implicit_reasoners

# Install dependencies
pip install simpletransformers pandas --quiet 2>/dev/null || echo "Dependencies already installed"

# Run training on minimal composition dataset
# Based on paper's configuration but with smaller scale for faster iteration
# Using encoder-decoder-name for GPT-2
python main.py \
    --data_dir=data/composition_minimal \
    --model_type=gpt2 \
    --encoder_decoder_type=gpt2 \
    --encoder_decoder_name=gpt2 \
    --init_weights \
    --n_layer=4 \
    --n_head=4 \
    --add_tokens \
    --no_dropout \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir \
    --save_best_model \
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
    --manual_seed=42

echo "=========================================="
echo "Training complete!"
echo "=========================================="

