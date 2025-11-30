# Deep Networks Always Grok and Here is Why

**Authors:** Ahmed Imtiaz Humayun, Randall Balestriero, Richard Baraniuk

**Paper:** [arXiv:2402.15555](https://arxiv.org/abs/2402.15555)

**Code:** https://bit.ly/grok-adversarial

## Summary

This paper demonstrates that grokking is widespread and occurs in practical settings, not just controlled algorithmic tasks. The authors introduce "delayed robustness"—where DNNs grok adversarial examples long after generalizing on clean test data. They explain grokking through "local complexity" of the DNN's spline partition.

**Key Contribution:** The paper shows that deep networks **always grok** when properly trained, and this phenomenon extends to adversarial robustness, not just clean accuracy.

## Datasets

- **MNIST:** 1,000 training samples
- **CIFAR-10:** Standard dataset
- **CIFAR-100:** Standard dataset
- **Imagenette:** Subset of ImageNet with 10 classes
- **Adversarial examples:** ℓ∞-PGD with various ε values

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
- **Training steps:** Extended (10^5 steps typical)

## Setup and Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

### Complete Replication with Adversarial Testing

**MNIST + MLP (with adversarial robustness):**
```bash
cd scripts
sbatch run_mnist_mlp_adversarial.sh
```

**CIFAR-10 + CNN (with adversarial robustness):**
```bash
cd scripts
sbatch run_cifar10_cnn.sh
```

**Imagenette + ResNet-18 (with adversarial robustness):**
```bash
cd scripts
sbatch run_imagenette_resnet.sh
```

### Local Execution with Adversarial Testing

**MNIST with adversarial robustness:**
```bash
python train.py \
    --model=mlp \
    --dataset=mnist \
    --train_size=1000 \
    --batch_size=200 \
    --lr=0.001 \
    --weight_decay=0.01 \
    --n_epochs=100000 \
    --device=cuda \
    --enable_adversarial \
    --adv_eval_batches=20
```

**CIFAR-10 with CNN:**
```bash
python train.py \
    --model=cnn \
    --dataset=cifar10 \
    --train_size=5000 \
    --batch_size=128 \
    --lr=0.001 \
    --weight_decay=0.0 \
    --n_epochs=100000 \
    --device=cuda \
    --enable_adversarial
```

**Imagenette with ResNet-18:**
```bash
python train.py \
    --model=resnet18 \
    --dataset=imagenette \
    --train_size=5000 \
    --batch_size=64 \
    --lr=0.001 \
    --weight_decay=0.0 \
    --n_epochs=100000 \
    --device=cuda \
    --enable_adversarial \
    --adv_eval_batches=10
```

### Adversarial Testing Options

```bash
--enable_adversarial          # Enable adversarial robustness testing
--adv_epsilons 0.06 0.10      # Custom epsilon values (default: [0.06, 0.10, 0.13, 0.16, 0.20])
--adv_eval_batches 20         # Number of batches for adversarial eval (speed vs accuracy)
```

## Expected Results

### Clean Accuracy (Grokking)
- Training accuracy reaches 100% (memorization)
- Test accuracy improves suddenly after memorization
- MNIST: ~89% test accuracy from only 1,000 samples
- Pattern: Train → 100%, Test → rapid jump, then plateau

### Adversarial Robustness (Delayed Robustness)
- Adversarial accuracy starts very low
- Improves gradually over many thousands of epochs
- **Key Finding:** Improvement continues long after clean accuracy plateaus
- Pattern: Clean accuracy plateaus → Continued training → Adversarial accuracy improves
- This demonstrates "delayed robustness" - the paper's main contribution

### Epsilon Dependence
- ε=0.06 (weakest attack): Highest adversarial accuracy
- ε=0.20 (strongest attack): Lowest adversarial accuracy
- All epsilon values show delayed improvement

### Universality
- Phenomenon observed across MNIST, CIFAR-10, Imagenette
- Works with MLP, CNN, and ResNet architectures
- Validates paper's title: "Deep Networks **Always** Grok"

## Output Files

- `results/*/training_history.json`: Training metrics including adversarial accuracies
- `results/*/checkpoints/`: Model checkpoints
- `data/`: Downloaded datasets (MNIST, CIFAR-10, CIFAR-100, Imagenette)

### Training History Format

```json
{
  "epoch": [0, 100, 200, ...],
  "train_loss": [...],
  "train_acc": [...],
  "test_loss": [...],
  "test_acc": [...],
  "adv_acc_eps_0.06": [...],
  "adv_acc_eps_0.10": [...],
  "adv_acc_eps_0.13": [...],
  "adv_acc_eps_0.16": [...],
  "adv_acc_eps_0.20": [...]
}
```

## Visualization

Generate plots showing delayed robustness:

```bash
cd ../  # Go to Replications directory
python plot_paper06_adversarial.py
```

This creates:
- Individual plots for each experiment showing clean and adversarial accuracies
- Comparison plots across all experiments
- Saved to `analysis_results/paper_06_*.png`

## Citation

```bibtex
@article{humayun2024deep,
  title={Deep Networks Always Grok and Here is Why},
  author={Humayun, Ahmed Imtiaz and Balestriero, Randall and Baraniuk, Richard},
  journal={arXiv preprint arXiv:2402.15555},
  year={2024}
}
```
