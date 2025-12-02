# Test Suite

This directory contains test scripts for validating the AGOP experiments infrastructure before running full-scale experiments. Tests verify import dependencies, model instantiation, data loading, training loops, and output serialization.

---

## Test Files

| File | Purpose | Runtime |
|------|---------|---------|
| `test_all_experiments.py` | Comprehensive test of all experiment configurations | ~10-30 min |
| `test_onehot_complete.py` | Validate one-hot encoding and model compatibility | ~1 min |
| `verify_imports.py` | Check all required imports are available | ~5 sec |
| `test_agop_slurm.sh` | SLURM-based single experiment test | ~5 min |
| `test_quick_train.sh` | Quick local training validation | ~2 min |
| `test_single_nanda.sh` | Single Nanda experiment test | ~5 min |

---

## Quick Start

### 1. Verify Imports

```bash
cd tests/
python verify_imports.py
```

Expected output:
```
Checking imports...
✓ torch
✓ numpy
✓ h5py
✓ tqdm
✓ yaml
All imports successful!
```

### 2. Test One-Hot Infrastructure

```bash
python test_onehot_complete.py
```

Validates:
- One-hot dataset creation (modular, MNIST, composition)
- Model forward passes with one-hot inputs
- Correct tensor shapes and dtypes

### 3. Run Quick Training Test

```bash
./test_quick_train.sh
```

Runs a minimal training loop (100 epochs) to verify:
- Training script execution
- Loss computation
- Metric logging
- File output

### 4. Comprehensive Test Suite

```bash
python test_all_experiments.py --all
```

Runs mini versions of all experiments across all datasets and optimizers.

---

## test_all_experiments.py

Comprehensive testing script that runs abbreviated versions of all experiment configurations.

### Test Configurations

Each dataset is tested with reduced parameters for fast execution:

| Dataset | Test Epochs | Test p/Points | AGOP Freq |
|---------|-------------|---------------|-----------|
| Nanda | 200 | p=97 | 50 |
| Softmax | 200 | p=97 | 50 |
| MNIST | 200 | 500 points | 50 |
| Composition | 200 | 200 facts | 50 |

### Usage

```bash
# Test all datasets with all optimizers
python test_all_experiments.py --all

# Test specific dataset
python test_all_experiments.py --dataset nanda

# Test specific optimizer
python test_all_experiments.py --optimizer adamw

# Test specific combination
python test_all_experiments.py --dataset nanda --optimizer muon

# Use CPU (recommended for testing)
python test_all_experiments.py --all --device cpu
```

### Command-Line Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--dataset` | nanda, softmax, mnist, composition, all | all | Dataset to test |
| `--optimizer` | adamw, muon, sgd, all | all | Optimizer to test |
| `--device` | cpu, cuda | cpu | Compute device |
| `--seed` | int | 42 | Random seed |
| `--all` | flag | — | Run all tests |

### Test Matrix

Full test suite runs 12 combinations:

| Dataset | AdamW | Muon | SGD |
|---------|-------|------|-----|
| Nanda | ✓ | ✓ | ✓ |
| Softmax | ✓ | ✓ | ✓ |
| MNIST | ✓ | ✓ | ✓ |
| Composition | ✓ | ✓ | ✓ |

### Output Verification

For each test, the script verifies:
1. Training script runs without errors
2. Output directory is created
3. Required files exist:
   - `config.json` — Experiment configuration
   - `training_history.json` — Training metrics
   - `agop_metrics.h5` — AGOP measurements
4. Files contain valid data

### Test Results

Results are saved to `test_results/test_summary_YYYYMMDD_HHMMSS.json`:

```json
{
  "summary": {
    "total_tests": 12,
    "successes": 12,
    "failures": 0,
    "total_time": 180.5
  },
  "results": {
    "nanda": {
      "adamw": {"success": true, "time": 15.2, ...},
      "muon": {"success": true, "time": 14.8, ...},
      ...
    },
    ...
  }
}
```

---

## verify_imports.py

Lightweight script to check all required dependencies are installed.

### Checked Modules

| Module | Purpose | Required |
|--------|---------|----------|
| torch | Deep learning framework | Yes |
| numpy | Numerical operations | Yes |
| h5py | HDF5 file I/O | Yes |
| tqdm | Progress bars | Yes |
| yaml | Config file parsing | Yes |
| torchvision | MNIST dataset | For MNIST |
| matplotlib | Plotting | For analysis |
| seaborn | Statistical plots | For analysis |
| scipy | Statistical tests | For analysis |

### Usage

```bash
python verify_imports.py
```

Exit code 0 indicates success; non-zero indicates missing dependencies.

---

## test_onehot_complete.py

Validates the one-hot encoding infrastructure and model compatibility.

### Tests Performed

1. **Dataset Creation**
   - `create_onehot_modular_dataset(p=97)`
   - `create_onehot_mnist_dataset(train_points=100)`
   - `create_onehot_composition_dataset(vocab_size=50)`

2. **Shape Verification**
   - Input tensor dimensions match expected [batch, input_dim]
   - Output tensor dimensions match expected [batch, num_classes]
   - Data types are float32 (continuous)

3. **One-Hot Properties**
   - Sum of each input vector equals expected (2 for modular, seq_len for composition)
   - Values are in {0, 1} for one-hot positions

4. **Model Forward Passes**
   - `ModularArithmeticMLP` with one-hot inputs
   - `OneHotReLUTransformer` with one-hot inputs
   - `OneHotStandardTransformer` with one-hot inputs
   - `MNISTModel` with continuous inputs

### Usage

```bash
python test_onehot_complete.py
```

Expected output:
```
Testing One-Hot Dataset Creation
================================
1. Modular arithmetic (p=97): ✓
   Shape: [2822, 194], Sum: 2.0
2. MNIST: ✓
   Shape: [100, 784], Range: [0, 1]
3. Composition: ✓
   Shape: [150, 500], Sum: 10.0

Testing Model Forward Passes
============================
1. ModularArithmeticMLP: ✓
2. OneHotReLUTransformer: ✓
3. OneHotStandardTransformer: ✓

All tests passed!
```

---

## Shell Scripts

### test_agop_slurm.sh

Tests a single experiment via SLURM submission.

```bash
sbatch test_agop_slurm.sh
```

Configuration:
- Single GPU job
- 1 hour time limit
- Nanda dataset with AdamW
- 500 epochs

### test_quick_train.sh

Local quick training test without SLURM.

```bash
./test_quick_train.sh
```

Runs Nanda experiment with:
- 100 epochs
- CPU device
- Reduced AGOP frequency

### test_single_nanda.sh

Single Nanda experiment test with full AGOP tracking.

```bash
./test_single_nanda.sh
```

---

## Test Results Directory

Test outputs are saved to `test_results/`:

```
test_results/
├── test_nanda/
│   ├── test_nanda_adamw_seed42/
│   ├── test_nanda_muon_seed42/
│   └── test_nanda_sgd_seed42/
├── test_softmax/
│   └── ...
├── test_mnist/
│   └── ...
├── test_composition/
│   └── ...
└── test_summary_YYYYMMDD_HHMMSS.json
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependency | Run `verify_imports.py`, install missing packages |
| CUDA errors | GPU not available | Use `--device cpu` |
| Timeout | Test took too long | Reduce `n_epochs` in test config |
| File not found | Wrong working directory | Run from `tests/` directory |

### Debug Mode

For verbose output during testing:

```bash
# Run single test with full output
python -c "
from test_all_experiments import run_test
result = run_test('nanda', 'adamw', device='cpu')
print(result)
"
```

### Memory Issues

If tests fail due to memory:

```bash
# Reduce batch processing
python test_all_experiments.py --dataset mnist --device cpu
```

---

## Continuous Integration

### Pre-Submission Checklist

Before submitting full experiments:

1. ✓ `python verify_imports.py` — All imports pass
2. ✓ `python test_onehot_complete.py` — Data/models work
3. ✓ `python test_all_experiments.py --dataset nanda --optimizer adamw` — Single test passes
4. ✓ `./test_quick_train.sh` — Training loop works

### Expected Test Duration

| Test | Duration | Memory |
|------|----------|--------|
| verify_imports | 5 sec | <100 MB |
| test_onehot_complete | 1 min | <1 GB |
| Single experiment test | 2-5 min | <2 GB |
| Full test suite | 20-30 min | <4 GB |

---

## Adding New Tests

To add tests for new experiment types:

1. Add configuration to `TEST_CONFIGS` in `test_all_experiments.py`:

```python
TEST_CONFIGS['new_dataset'] = {
    'script': 'train_new_dataset_agop.py',
    'args': {
        'n_epochs': 200,
        'agop_freq': 50,
        # ... other reduced parameters
    },
}
```

2. Update `verify_imports.py` if new dependencies are required.

3. Add dataset-specific validation to `test_onehot_complete.py`.

---

*Last Updated: December 2024*

