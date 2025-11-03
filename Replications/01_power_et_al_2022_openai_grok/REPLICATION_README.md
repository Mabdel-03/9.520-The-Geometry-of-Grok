# Replication Guide: OpenAI Grok

This directory contains the original OpenAI Grok repository for replicating the seminal grokking paper by Power et al. (2022).

## Original Paper

**Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets**

- **Authors:** Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra
- **Paper:** [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- **Original Repo:** https://github.com/openai/grok

## Quick Start

### Installation

```bash
pip install -e .
```

### Running with SLURM (HPC)

```bash
chmod +x run_modular_addition.sh
sbatch run_modular_addition.sh
```

### Running Locally

```bash
./scripts/train.py
```

## Key Experiments to Replicate

### 1. Modular Addition (Main Result)
```bash
python scripts/train.py \
    --operation=x+y \
    --prime=97 \
    --training_fraction=0.5 \
    --weight_decay=1
```

### 2. Modular Subtraction
```bash
python scripts/train.py \
    --operation=x-y \
    --prime=97 \
    --training_fraction=0.5
```

### 3. Modular Division
```bash
python scripts/train.py \
    --operation=x/y \
    --prime=97 \
    --training_fraction=0.5
```

### 4. Varying Training Fraction
```bash
for frac in 0.1 0.3 0.5 0.7 0.9; do
    python scripts/train.py --training_fraction=$frac
done
```

## Expected Results

- Train accuracy reaches 100% quickly
- Test accuracy remains low for extended period
- Sudden jump in test accuracy (grokking) after 10k-100k steps
- Weight decay is critical for observing grokking

## For Gradient Outer Product Analysis

The codebase logs model checkpoints regularly. To compute gradient outer products:

1. Load checkpoints from different training stages
2. Compute gradients on train/test batches
3. Compute outer products and track their evolution

See the original README.md for more details on the codebase structure.

## Citation

```bibtex
@article{power2022grokking,
  title={Grokking: Generalization beyond overfitting on small algorithmic datasets},
  author={Power, Alethea and Burda, Yuri and Edwards, Harri and Babuschkin, Igor and Misra, Vedant},
  journal={arXiv preprint arXiv:2201.02177},
  year={2022}
}
```

