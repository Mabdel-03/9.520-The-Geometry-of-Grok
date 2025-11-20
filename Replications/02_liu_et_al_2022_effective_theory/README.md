# Liu et al. (2022): Towards Understanding Grokking

## Paper Information

**Title:** Towards Understanding Grokking: An Effective Theory of Representation Learning

**Authors:** Ziming Liu, Ouail Kitouni, Niklas Nolte, Eric J. Michaud, Max Tegmark, Mike Williams

**Publication:** arXiv:2205.10343 (2022)

**Links:**
- Paper: https://arxiv.org/abs/2205.10343
- Original Repository: https://github.com/ejmichaud/grokking-squared

## Key Contributions

This paper provides a theoretical framework for understanding the grokking phenomenon through:

1. **Representation Quality Index (RQI)**: A novel metric to quantify the structure of learned representations
2. **Phase Diagram**: Characterizes four learning phases (comprehension, grokking, memorization, confusion)
3. **Effective Theory**: Mathematical framework explaining grokking dynamics through representation learning
4. **Parallelogram Structure**: Shows how modular arithmetic creates geometric constraints that enable generalization

## Architecture

### Encoder-Decoder Model

The model consists of two components that learn separately:

**Encoder:**
- Maps input symbols to internal representations
- Architecture: 2-3 layer MLP
- Width: 200 units per layer
- Activation: Tanh
- Output: 1D representation (reprs_dim=1)

**Decoder:**
- Maps combined representations to output
- Architecture: 2-3 layer MLP
- Width: 200 units per layer
- Activation: Tanh
- Operation: Dec(E(a) + E(b)) for addition task

**Key Design:**
- Encoder learns sparse, structured representations of symbols
- Addition performed exactly in representation space
- Decoder maps representations to outputs
- Separate optimizers with different learning rates

## Dataset and Task

### Modular Addition

**Task:** Learn the operation (a + b) mod p

**Dataset Generation:**
- Modulus: p = 10
- All possible pairs: p × (p+1) / 2 = 55 samples (since a+b = b+a)
- Training set: 45 samples (~82%)
- Test set: 10 samples (~18%)
- Random split with fixed seed

**Input/Output:**
- Input: Two symbols (a, b)
- Output: One symbol (a+b mod p)
- Representation: One-hot encoding or learned embeddings

**Parallelogram Structure:**
- Key geometric property: If (a,b) and (c,d) satisfy a+b = c+d, then E(a)+E(b) = E(c)+E(d)
- RQI measures how many such parallelograms are satisfied

## Training Configuration

### Hyperparameters

```python
# Model
p = 10                      # Modulus
reprs_dim = 1              # 1D internal representation
width = 200                # Hidden layer width
depth = 2-3                # Number of layers

# Dataset
train_num = 45             # Training samples (out of 55 total)
seed = 58                  # Random seed

# Training
steps = 5000               # Training iterations
batch_size = 45            # Full batch
loss_type = "MSE"          # Mean squared error

# Optimization
eta_reprs = 1e-3          # Learning rate for representations (10x larger!)
eta_dec = 1e-4            # Learning rate for decoder
weight_decay_reprs = 0.0  # NO weight decay for representations
weight_decay_dec = 0.0    # NO weight decay for decoder
optimizer = "AdamW"       # Adam with weight decay support
```

### Key Training Insights

1. **Differential Learning Rates**: Representations learn 10× faster than decoder
2. **No Weight Decay**: Unlike other grokking papers, this architecture doesn't require weight decay
3. **Full Batch Training**: Uses all training samples in each step
4. **Three-Stage Learning**: RQI improves → Train acc improves → Test acc improves

## Expected Results (From Paper)

### Grokking Phenomenon

- **Training accuracy** reaches 90% at: ~1000-1500 steps
- **Test accuracy** reaches 90% at: ~1500-2000 steps
- **Grokking delay**: 300-500 steps (delayed generalization)
- **Final performance**: 100% train, 100% test

### RQI Trajectory

- **RQI threshold (0.95)** reached at: ~800 steps (before accuracy improves!)
- Shows representations develop structure before generalization
- Final RQI: 1.0 (perfect parallelogram structure)

### Four Learning Phases

Based on (learning rate, weight decay) sweeps:

1. **Comprehension** (green): Fast train & test learning, small delay
2. **Grokking** (yellow): Fast train learning, delayed test learning
3. **Memorization** (purple): Fast train, test never improves
4. **Confusion** (black): Neither train nor test improve

## Running on This Cluster

### Quick Start

```bash
# Submit main experiment
sbatch run_experiment.sh

# Or run directly
python run_experiment.py --experiment toy_model
```

### Available Experiments

**Experiment 1: Toy Model (Main Grokking Demo)**
```bash
python run_experiment.py --experiment toy_model \
    --p 10 \
    --train_num 45 \
    --steps 5000 \
    --eta_reprs 1e-3 \
    --eta_dec 1e-4
```

**Experiment 2: Phase Diagram Sweep**
```bash
python run_experiment.py --experiment phase_diagram \
    --lr_range "1e-4,1e-3,1e-2" \
    --wd_range "0,1,5,10"
```

### Resource Requirements

- **GPU**: 1× A100 (or any CUDA GPU)
- **Memory**: 8 GB
- **Time**: ~30 seconds for toy model, ~2 hours for full phase diagram
- **CPUs**: 2

## Results Location

All experimental results are stored in the `results/` directory:

- `results/experiment_1_toy_model/` - Main grokking demonstration
- `results/experiment_2_phase_diagram/` - Phase diagram data

Each subdirectory contains:
- `README.md` - Detailed experiment documentation
- Raw data files (JSON, TXT, HDF5)
- Summary metrics

## Verification Checklist

To verify successful replication:

- ✅ Training runs without errors
- ✅ Train accuracy reaches 90%+ 
- ✅ Test accuracy reaches 90%+ after train accuracy
- ✅ Grokking delay is 300-500 steps
- ✅ Final performance is 100%/100%
- ✅ RQI reaches 1.0
- ✅ RQI improves before accuracy

## Citation

```bibtex
@article{liu2022towards,
  title={Towards Understanding Grokking: An Effective Theory of Representation Learning},
  author={Liu, Ziming and Kitouni, Ouail and Nolte, Niklas and Michaud, Eric J and Tegmark, Max and Williams, Mike},
  journal={arXiv preprint arXiv:2205.10343},
  year={2022}
}
```

## Notes

- This replication uses the **original code** from the paper's repository
- The core training algorithm (`train_add.py`) is unchanged from the original
- Results should closely match the paper's findings
- For questions or issues, refer to the original repository or paper

