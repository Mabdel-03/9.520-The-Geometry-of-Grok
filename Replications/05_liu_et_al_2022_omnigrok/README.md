# Omnigrok
This is the code repo for the paper: ["Omnigrok: Grokking Beyond Algorithmic Data"](https://openreview.net/forum?id=zDiHoIWa0q1), accepted in ICLR 2023 as spotlight. We elucidate [the grokking phenomenon](https://arxiv.org/abs/2201.02177) from the perspective of loss landscapes, and show that grokking can not only happen for algorithmic datasets and toy teacher-students setups, but also for standard machine learning datasets (e.g., MNIST, IMDb movie reviews, QM9 molecules).

The examples used in this paper are relatively small-scale. We make our codes as minimal as possible: each example is self-consistent, kept in a single folder. 
|Examples| Figure in [paper](https://openreview.net/forum?id=zDiHoIWa0q1) | Folder |
|--|--|--|
|Teacher-student|Figure 2| ./teacher-student|
| MNIST handwritten digits | Figure 3 | ./mnist |
| IMDb Movie Reviews | Figure 4 | ./imdb |
| QM9 Molecule properties  | Figure 5 | ./qm9 |
| Modular addition | Figure 6 & 8 | ./mod-addition |
| MNIST Representation | Figure 7 | ./mnist-repr |

For each example, we conduct two kinds of experiments: 
* (1) reduced landscape analysis: the weight norm is fixed during training. 
* (2) grokking experiments: the weight norm is not fixed during training (standard training).

Each folder (except for MNIST representation) contains two subfolders, for (1) "landscape" and (2) "grokking".

---

# Replication Guide: Omnigrok - Grokking Beyond Algorithmic Data

This directory contains the Omnigrok repository extending grokking to diverse datasets.

## Original Paper

**Omnigrok: Grokking Beyond Algorithmic Data**

- **Authors:** Ziming Liu, Eric J. Michaud, Max Tegmark
- **Paper:** [arXiv:2210.01117](https://arxiv.org/abs/2210.01117)
- **Original Repo:** https://github.com/KindXiaoming/Omnigrok

## Key Contributions

- Extends grokking to images (MNIST), text (IMDb), molecules (QM9)
- Introduces "LU mechanism" (L-shaped train loss, U-shaped test loss vs weight norm)
- Shows grokking is universal, not specific to algorithmic tasks

## Quick Start

### Installation

```bash
pip install torch torchvision numpy matplotlib jupyter
```

### Running with SLURM

```bash
chmod +x run_mnist.sh
sbatch run_mnist.sh
```

## Key Experiments

### 1. MNIST Grokking
```bash
cd mnist/grokking
jupyter notebook mnist-grokking.ipynb
```

Or convert and run:
```bash
jupyter nbconvert --to script mnist-grokking.ipynb
python mnist-grokking.py
```

### 2. Modular Addition
```bash
cd mod-addition/grokking
jupyter notebook modular-addition-grokking.ipynb
```

### 3. IMDb Sentiment
```bash
cd imdb/grokking
# Follow notebook instructions
```

### 4. QM9 Molecular Properties
```bash
cd qm9/grokking
jupyter notebook qm9-grokking.ipynb
```

## Repository Structure

Each dataset has two subdirectories:
- `grokking/`: Demonstrates grokking phenomenon with reduced training data
- `landscape/`: Analyzes loss landscape and LU mechanism

## Key Hyperparameters

### MNIST
- Training size: 1,000 (reduced from 60,000)
- Architecture: 3-layer MLP, width 200
- Weight decay: 0.01
- Learning rate: 0.001

### IMDb
- Training size: 1,000
- Architecture: 2-layer LSTM
- Embedding dim: 64, Hidden dim: 128

### QM9
- Training size: 100-3000
- Architecture: 2-layer GCNN

## Expected Results

- **MNIST:** Grokking occurs around 10k-50k epochs with 1k training samples
- **IMDb:** More subtle grokking, longer training required
- **QM9:** Clear grokking with very small training sets (100-1000)
- All show LU mechanism: train loss L-shape, test loss U-shape vs weight norm

## Important Notes

- All experiments use **reduced training sets** to induce grokking
- Notebooks are self-contained but may need dataset downloads
- Some experiments are computationally intensive

## For Gradient Analysis

This repository is valuable because it shows grokking across diverse domains:
- Compare gradient dynamics across different data modalities
- Investigate if LU mechanism has universal gradient signatures
- Study representation learning in non-algorithmic tasks

## Citation

```bibtex
@article{liu2022omnigrok,
  title={Omnigrok: Grokking beyond algorithmic data},
  author={Liu, Ziming and Michaud, Eric J and Tegmark, Max},
  journal={arXiv preprint arXiv:2210.01117},
  year={2022}
}
```
