# Progress Measures for Grokking via Mechanistic Interpretability

**Authors:** Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt

**Paper:** [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)

**Website:** https://neelnanda.io/grokking-paper

## Summary

This paper reverse-engineers the algorithm learned by transformers on modular addition to understand grokking mechanistically. The authors discover that models implement a "Fourier multiplication algorithm"—mapping inputs to circles using discrete Fourier transforms and trigonometric identities. They identify three continuous training phases: memorization, circuit formation, and cleanup.

## Dataset

- **Task:** Modular addition $(a + b) \mod P$ where $P = 113$ (prime)
- **Input format:** "a b =" where $a, b$ are represented as tokens
- **Training data:** 30% of all possible pairs (113 × 113)
- **Test data:** All remaining pairs

## Model Architecture

- **Type:** 1-layer ReLU Transformer
- **Model dimension:** $d = 128$
- **Attention heads:** 4 heads, dimension $d/4 = 32$ each
- **MLP hidden units:** 512
- **No LayerNorm**
- Output read from position above "=" token

## Training Hyperparameters

- **Optimizer:** AdamW
- **Learning rate:** 0.001
- **Weight decay:** 1.0
- **Batch size:** Full batch gradient descent
- **Training epochs:** 40,000
- **Loss function:** Cross-entropy

## Setup and Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch numpy matplotlib
```

## Running Experiments

### Using SLURM (on HPC cluster)

```bash
# Make script executable
chmod +x run_modular_addition.sh

# Submit job
sbatch run_modular_addition.sh
```

### Running Locally

```bash
python train.py \
    --p=113 \
    --train_fraction=0.3 \
    --d_model=128 \
    --n_heads=4 \
    --d_mlp=512 \
    --lr=0.001 \
    --weight_decay=1.0 \
    --n_epochs=40000 \
    --device=cuda
```

## Expected Results

- **Train accuracy:** Should reach ~100% within first few thousand steps
- **Test accuracy:** Should remain low for extended period, then suddenly jump to ~100% (grokking)
- **Grokking point:** Typically occurs around epochs 10,000-30,000

## Output Files

- `logs/training_history.json`: Training and test metrics over time
- `checkpoints/checkpoint_epoch_*.pt`: Model checkpoints at regular intervals

## Notes for Gradient Outer Product Analysis

This implementation logs all necessary metrics to track the grokking phenomenon. For gradient outer product analysis, you can modify `train.py` to compute and save gradient information at regular intervals.

## Citation

```bibtex
@article{nanda2023progress,
  title={Progress Measures for Grokking via Mechanistic Interpretability},
  author={Nanda, Neel and Chan, Lawrence and Lieberum, Tom and Smith, Jess and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2301.05217},
  year={2023}
}
```

