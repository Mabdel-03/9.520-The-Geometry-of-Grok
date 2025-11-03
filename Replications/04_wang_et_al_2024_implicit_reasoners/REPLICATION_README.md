# Replication Guide: Grokked Transformers are Implicit Reasoners

This directory contains the GrokkedTransformer repository for investigating compositional reasoning and grokking.

## Original Paper

**Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization**

- **Authors:** Boshi Wang, Xiang Yue, Yu Su, Huan Sun
- **Paper:** [arXiv:2405.15071](https://arxiv.org/abs/2405.15071)
- **Original Repo:** https://github.com/OSU-NLP-Group/GrokkedTransformer

## Key Contributions

- Demonstrates grokking on knowledge graph reasoning tasks
- Studies composition (multi-hop) and comparison reasoning
- Mechanistic analysis of different generalizing circuits
- Shows systematicity depends on circuit architecture

## Quick Start

### Installation

```bash
pip install -r requirements.txt
# The repository includes transformers library and custom modifications
```

### Running with SLURM

```bash
chmod +x run_composition.sh
sbatch run_composition.sh
```

## Key Experiments

### 1. Composition Task (Two-Hop Reasoning)
```bash
python src/train.py \
    --task=composition \
    --num_entities=2000 \
    --batch_size=512 \
    --lr=1e-4 \
    --weight_decay=0.1 \
    --max_steps=2000000
```

### 2. Comparison Task
```bash
python src/train.py \
    --task=comparison \
    --num_entities=1000 \
    --batch_size=512
```

### 3. Varying Dataset Size
```bash
for ne in 2000 5000 10000; do
    python src/train.py --task=composition --num_entities=$ne
done
```

## Expected Results

- **Composition task:** Grokking occurs but OOD generalization fails
- **Comparison task:** Grokking occurs with successful OOD generalization
- Training time: Can take days/weeks depending on configuration
- Grokking point: Varies significantly (100k to 1M+ steps)

## Directory Structure

- `src/`: Main training code
- `transformers/`: Modified transformers library
- `data/`: Knowledge graph generation scripts
- `analysis/`: Mechanistic interpretability notebooks

## Important Notes

- Large-scale experiments require significant compute
- The paper uses GPT-2 style transformers (8 layers, 768 dim)
- Weight decay of 0.1 is critical for grokking
- Consider starting with smaller entity counts for testing

## For Gradient Analysis

This codebase is particularly interesting for gradient outer product analysis because:
1. It shows different circuits for different reasoning types
2. Grokking occurs on more complex tasks than modular arithmetic
3. Can compare in-distribution vs OOD generalization

## Citation

```bibtex
@article{wang2024grokked,
  title={Grokked transformers are implicit reasoners: A mechanistic journey to the edge of generalization},
  author={Wang, Boshi and Yue, Xiang and Su, Yu and Sun, Huan},
  journal={arXiv preprint arXiv:2405.15071},
  year={2024}
}
```

