# Getting Started with GOP Analysis Framework

This guide will help you get started with analyzing gradient outer products in grokking experiments.

## Installation

```bash
cd New_Explorations
pip install -r requirements.txt
```

## Quick Test (Recommended First Step)

Run validation tests to ensure framework is working:

```bash
cd framework
python validate.py
```

Expected output:
```
✓ GOP computation test PASSED
✓ Trace test PASSED
✓ Eigenvalue sum test PASSED
✓ Frobenius norm test PASSED
✓ Storage test PASSED
✓ Memory estimation test PASSED

🎉 All tests passed! Framework is ready to use.
```

## Your First GOP Experiment

Start with the **smallest model** (Linear Estimators - Levi et al.):

### Step 1: Review Configuration

```bash
cat configs/09_levi_linear.yaml
```

This shows:
- Model has ~1K parameters (very manageable)
- GOP will be ~4 MB per epoch
- Total storage: ~16 GB for 100K epochs

### Step 2: Run Locally (Test Mode - 100 epochs)

```bash
cd experiments/09_levi_linear

# Modify config to run only 100 epochs for testing
python wrapped_train.py --config ../../configs/09_levi_linear.yaml
```

This will take ~5-10 minutes and create:
```
../../results/09_levi_linear/
├── metrics.h5         # ~100 KB
├── gop_full.h5        # ~400 MB (100 epochs × 4 MB)
├── gop_layers.h5      # ~100 MB
└── config.json
```

### Step 3: Visualize Results

```bash
cd ../../analysis
python visualize_gop.py --results_dir ../results/09_levi_linear
```

This creates plots in `../results/09_levi_linear/plots/`:
- Training and test curves
- GOP metrics evolution
- Top eigenvalues over time
- Comprehensive summary

## Running on HPC

Once local testing works:

```bash
cd experiments/09_levi_linear

# Edit SLURM script for your cluster
nano run_gop_analysis.sh

# Submit job
sbatch run_gop_analysis.sh

# Monitor
squeue -u $USER
tail -f logs/gop_linear_*.out
```

## Next Steps: Medium-Sized Models

After linear estimators work, try:

### 1. Modular Polynomials (Doshi - ~50K params)

```bash
cd experiments/08_doshi_polynomials
sbatch run_gop_analysis.sh
```

Storage: ~50 GB for 50K epochs

### 2. Mechanistic Interpretability (Nanda - ~100K params)

```bash
cd experiments/03_nanda_progress
sbatch run_gop_analysis.sh
```

Storage: ~160 GB for 40K epochs

### 3. Slingshot Mechanism (Thilak - ~150K params)

```bash
cd experiments/07_thilak_slingshot
sbatch run_gop_analysis.sh
```

Storage: ~360 GB for 50K epochs (be sure you have space!)

## Storage Management

### Check Available Space

Before running:
```bash
df -h $HOME  # Check home directory space
```

### Monitor During Training

```bash
watch -n 60 du -sh New_Explorations/results/
```

### Reduce Storage Requirements

If running low on space, edit config to:

```yaml
gop_tracking:
  frequency: 10  # Track every 10 epochs instead of 1

storage:
  store_full_gop: false  # Store only metrics, not full matrices
  compression_level: 9    # Maximum compression
```

## Analyzing Your Results

### Basic Analysis

```bash
cd analysis

# Visualize single experiment
python visualize_gop.py --results_dir ../results/09_levi_linear

# Compare multiple experiments
python compare_experiments.py \
    --experiments ../results/09_levi_linear ../results/08_doshi_polynomials \
    --save_dir comparison_plots/
```

### Custom Analysis in Python

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load metrics
with h5py.File('results/09_levi_linear/metrics.h5', 'r') as f:
    test_acc = f['test_acc'][:]
    gop_trace = f['gop_trace'][:]
    eigenvalues_top_k = f['eigenvalues_top_k'][:]

# Plot eigenvalue spectrum evolution
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(eigenvalues_top_k[:, i], label=f'λ_{i+1}')
plt.xlabel('Epoch')
plt.ylabel('Eigenvalue')
plt.yscale('log')
plt.legend()
plt.title('Top-10 Eigenvalues During Grokking')
plt.show()

# Correlate with grokking
grok_epoch = np.where(test_acc > 0.9)[0][0] if np.any(test_acc > 0.9) else None
if grok_epoch:
    print(f"Grokking at epoch {grok_epoch}")
    print(f"Trace at grokking: {gop_trace[grok_epoch]:.2e}")
    print(f"Top eigenvalue: {eigenvalues_top_k[grok_epoch, 0]:.2e}")
```

### Load Full GOP Matrix for Specific Epoch

```python
import h5py

# Load full GOP for epoch 10000
with h5py.File('results/09_levi_linear/gop_full.h5', 'r') as f:
    gop = f['epoch_10000']['gop'][:]
    eigenvalues = f['epoch_10000']['eigenvalues'][:]
    eigenvectors_top_k = f['epoch_10000']['eigenvectors_top_k'][:]

print(f"GOP shape: {gop.shape}")
print(f"Number of eigenvalues: {len(eigenvalues)}")
print(f"Top-k eigenvectors shape: {eigenvectors_top_k.shape}")
```

## Creating Wrappers for Remaining Experiments

For papers 01, 02, 04, 05, 10 (cloned repos), use the template:

### Step 1: Copy Template

```bash
cp framework/template_wrapper.py experiments/01_power_et_al/wrapped_train.py
```

### Step 2: Follow TODO Comments

Edit `wrapped_train.py` and:
1. Import the original model code
2. Replace dataset creation with actual code
3. Replace model initialization
4. Replace optimizer and loss function
5. Fill in training and evaluation logic

### Step 3: Create Config YAML

Copy and adapt an existing config:
```bash
cp configs/03_nanda_progress.yaml configs/01_power_grok.yaml
# Edit with paper-specific parameters
```

### Step 4: Create SLURM Script

```bash
cp experiments/03_nanda_progress/run_gop_analysis.sh experiments/01_power_et_al/
# Edit job name and parameters
```

## Troubleshooting

### "Out of memory" Error

**For GOP computation:**
```yaml
gop_tracking:
  use_gpu: false  # Move GOP to CPU
```

**For training:**
- Reduce batch size in original training params
- Use smaller model

### "Disk quota exceeded"

**Solutions:**
1. Store only metrics:
   ```yaml
   storage:
     store_full_gop: false
   ```

2. Increase tracking frequency:
   ```yaml
   gop_tracking:
     frequency: 100  # Every 100 epochs
   ```

3. Clean old results:
   ```bash
   rm -rf results/old_experiment/
   ```

### Slow Training

**If eigenvalue computation is bottleneck:**

Modify `framework/gop_metrics.py`:
```python
# Use only top-k eigenvalues (much faster)
eigenvalues, _ = self.compute_eigenvalues(gop_matrix, top_k_only=True)
```

Or reduce frequency:
```yaml
gop_tracking:
  frequency: 10
```

## Expected Workflow

1. ✅ **Validate:** Run `framework/validate.py`
2. ✅ **Test small:** Run linear estimators for 100 epochs
3. ✅ **Verify:** Check that HDF5 files are created and loadable
4. ✅ **Visualize:** Run `visualize_gop.py` on test results
5. ✅ **Scale up:** Run full experiments on HPC
6. ✅ **Analyze:** Use analysis tools to study GOP dynamics

## Understanding the Output

### metrics.h5 Structure

```python
metrics.h5/
  train_loss: [n_epochs]           # Training loss over time
  test_loss: [n_epochs]            # Test loss
  train_acc: [n_epochs]            # Training accuracy
  test_acc: [n_epochs]             # Test accuracy (grokking curve!)
  gop_trace: [n_epochs]            # GOP trace
  gop_frobenius_norm: [n_epochs]   # GOP Frobenius norm
  gop_spectral_norm: [n_epochs]    # Largest eigenvalue
  gop_rank: [n_epochs]             # Effective rank
  gop_condition_number: [n_epochs] # Condition number
  eigenvalues_top_k: [n_epochs, k] # Top-k eigenvalues
  ... (more metrics)
```

### gop_full.h5 Structure

```python
gop_full.h5/
  epoch_0/
    gop: [M, M]                    # Full GOP matrix
    eigenvalues: [M]               # All eigenvalues
    eigenvectors_top_k: [M, k]     # Top-k eigenvectors
  epoch_100/
    ...
  ...
```

## Questions to Explore

With this framework, you can investigate:

1. **How does the eigenvalue spectrum evolve during grokking?**
   - Do eigenvalues change suddenly or gradually?
   - What happens to the rank during the transition?

2. **Are there GOP signatures that predict grokking?**
   - Does trace show patterns before test acc jumps?
   - Do certain eigenvalues grow/shrink predictably?

3. **How do different optimizers affect GOP dynamics?**
   - Compare Adam vs AdamW vs SGD
   - Slingshot mechanism in eigenvalue space?

4. **Do different datasets show similar GOP patterns?**
   - Compare modular arithmetic vs MNIST vs linear
   - Universal grokking signatures?

5. **What role does weight decay play in GOP evolution?**
   - How does it affect eigenvalue spectrum?
   - Relationship to rank and condition number?

## Getting Help

- Check `USAGE_GUIDE.md` for detailed usage
- See `README.md` for framework overview
- Review template in `framework/template_wrapper.py`
- Examine working examples: `experiments/03_nanda_progress/wrapped_train.py`

Good luck with your grokking research! 🚀

