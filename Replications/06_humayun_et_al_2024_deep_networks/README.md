# Deep Networks Always Grok and Here is Why

**Authors:** Ahmed Imtiaz Humayun, Randall Balestriero, Richard Baraniuk

**Paper:** [arXiv:2402.15555](https://arxiv.org/abs/2402.15555)

**Code:** https://bit.ly/grok-adversarial

## Summary

This paper demonstrates that grokking is widespread and occurs in practical settings, not just controlled algorithmic tasks. The authors introduce "delayed robustness"—where DNNs grok adversarial examples long after generalizing on clean test data. They explain grokking through "local complexity" of the DNN's spline partition.

## Datasets

- **MNIST:** 1,000 training samples
- **CIFAR-10:** Standard dataset
- **CIFAR-100:** Standard dataset
- **Imagenette:** Subset of ImageNet with 10 classes
- **Adversarial examples:** $\ell_\infty$-PGD with various $\epsilon$ values

## Model Architectures

### MNIST
- 4-layer ReLU MLP, width 200
- Varying depths (2-6) and widths (20-2000)

### CIFAR-10/100
- CNN with 5 convolutional layers, 2 linear layers
- ResNet-18 (pre-activation, width 16, no batch normalization)

## Training Hyperparameters

- **Optimizer:** Adam
- **Learning rate:** 0.001 (most experiments)
- **Weight decay:** 0 (most), 0.01 (MNIST-MLP)
- **Loss function:** Cross-entropy
- **Batch size:** 200 (MNIST), 64-512 (CIFAR)
- **Training steps:** Extended ($10^5$ steps typical)

## Setup and Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

### MNIST with MLP (SLURM)

```bash
chmod +x run_mnist_mlp.sh
sbatch run_mnist_mlp.sh
```

### MNIST Locally

```bash
python train.py \
    --model=mlp \
    --dataset=mnist \
    --train_size=1000 \
    --batch_size=200 \
    --lr=0.001 \
    --weight_decay=0.01 \
    --n_epochs=100000 \
    --device=cuda
```

### CIFAR-10 with ResNet-18

```bash
python train.py \
    --model=resnet18 \
    --dataset=cifar10 \
    --batch_size=64 \
    --lr=0.001 \
    --weight_decay=0.0 \
    --n_epochs=100000 \
    --device=cuda
```

## Expected Results

- Clean test accuracy reaches high performance relatively quickly
- Adversarial robustness continues to improve much later (delayed robustness)
- Demonstrates grokking beyond algorithmic tasks

## Output Files

- `logs/training_history.json`: Training metrics
- `checkpoints/checkpoint_epoch_*.pt`: Model checkpoints
- `data/`: Downloaded datasets

## Citation

```bibtex
@article{humayun2024deep,
  title={Deep Networks Always Grok and Here is Why},
  author={Humayun, Ahmed Imtiaz and Balestriero, Randall and Baraniuk, Richard},
  journal={arXiv preprint arXiv:2402.15555},
  year={2024}
}
```

