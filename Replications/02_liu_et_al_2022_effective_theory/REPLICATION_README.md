# Replication Guide: Effective Theory of Representation Learning

This directory contains the grokking-squared repository for replicating Liu et al.'s effective theory paper.

## Original Paper

**Towards Understanding Grokking: An Effective Theory of Representation Learning**

- **Authors:** Ziming Liu, Ouail Kitouni, Niklas Nolte, Eric J. Michaud, Max Tegmark, Mike Williams
- **Paper:** [arXiv:2205.10343](https://arxiv.org/abs/2205.10343)
- **Original Repo:** https://github.com/ejmichaud/grokking-squared

## Key Contributions

- Phase diagram of grokking (comprehension, grokking, memorization, confusion)
- Representation Quality Index (RQI) to quantify structure
- Theoretical framework for understanding grokking

## Quick Start

### Running with SLURM

```bash
chmod +x run_toy_model.sh
sbatch run_toy_model.sh
```

### Running Locally

#### Toy Model (Addition)
```bash
python scripts/run_toy_model.py \
    --num_train=45 \
    --lr_embed=1e-3 \
    --lr_decoder=1e-3 \
    --weight_decay=1.0
```

#### Permutation Groups
```bash
python scripts/run_toy_model_permutations_ng.py
```

## Key Experiments

### 1. Phase Diagram Sweep
Vary weight decay and learning rate to create phase diagram:
```bash
for wd in 0.0 0.1 0.5 1.0 5.0 10.0; do
    for lr in 1e-4 1e-3 1e-2; do
        python scripts/run_toy_model.py --weight_decay=$wd --lr_decoder=$lr
    done
done
```

### 2. Representation Quality Analysis
The repository includes notebooks for analyzing learned representations:
```bash
cd toy/
jupyter notebook Figure3.ipynb  # RQI analysis
```

## Directory Structure

- `scripts/`: Training scripts
- `toy/`: Toy model experiments and analysis notebooks
- `phase_diagram_plot/`: Phase diagram generation code

## Important Notes

- Embeddings trained with Adam (no weight decay)
- Decoder trained with AdamW (with weight decay)
- Different hyperparameters for embeddings vs decoder is key insight

## Citation

```bibtex
@article{liu2022towards,
  title={Towards understanding grokking: An effective theory of representation learning},
  author={Liu, Ziming and Kitouni, Ouail and Nolte, Niklas and Michaud, Eric and Tegmark, Max and Williams, Mike},
  journal={arXiv preprint arXiv:2205.10343},
  year={2022}
}
```

