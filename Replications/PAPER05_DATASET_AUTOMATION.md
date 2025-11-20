# Paper 05: Dataset Download Automation - Summary

**Question**: Can you programmatically do any of the necessary dataset downloads?

**Answer**: Yes! 5 out of 6 experiments download automatically. Only IMDb requires manual intervention.

---

## ✅ **Fully Automated Datasets** (5/6)

### 1. MNIST (Experiments 1 & 6)
```python
# Handled automatically by PyTorch
train = torchvision.datasets.MNIST(root=download_directory, 
                                   train=True, 
                                   transform=torchvision.transforms.ToTensor(), 
                                   download=True)  # ← Automatic download
```
- **Status**: ✅ **Fully automatic**
- **Size**: ~11 MB
- **Action needed**: None - downloads on first run

### 2. QM9 Molecules (Experiment 3)
```python
# Handled automatically by PyTorch Geometric
dset = QM9('.')  # ← Automatic download on first access
```
- **Status**: ✅ **Fully automatic**
- **Size**: Variable (molecular structure data)
- **Prerequisites**: `pip install torch-geometric torch-scatter torch-sparse`
- **Action needed**: None - script installs dependencies and downloads data

### 3. Teacher-Student (Experiment 4)
```python
# Synthetic data generated programmatically
inputs_train = torch.normal(0, 1, size=(train_size, d_in))
labels_train = teacher(inputs_train).detach()
```
- **Status**: ✅ **Synthetic (no download)**
- **Action needed**: None - data generated during execution

### 4. Modular Addition (Experiment 5)
```python
# Synthetic data generated programmatically
x, y = torch.meshgrid(torch.arange(p), torch.arange(p))
answers = ((x + y) % p)
```
- **Status**: ✅ **Synthetic (no download)**
- **Action needed**: None - data generated during execution

### 5. MNIST Representation (Experiment 6)
- **Status**: ✅ **Same as Experiment 1**
- **Action needed**: None - uses full MNIST dataset (auto-downloads)

---

## ⚠️ **Semi-Automated Dataset** (1/6)

### IMDb Sentiment Reviews (Experiment 2)

**Current Implementation:**
- **Status**: ⚠️ **Requires Kaggle API setup OR manual download**
- **Why not fully automatic?**: Kaggle requires authentication
- **Size**: ~65 MB

**Option A: Programmatic Download (Requires Setup)**

Created: `scripts/download_imdb_dataset.py`

**Usage:**
```bash
# 1. One-time setup
pip install kaggle

# 2. Get Kaggle API credentials
# Visit: https://www.kaggle.com/settings
# Click: "Create New API Token"
# Save: kaggle.json to ~/.kaggle/kaggle.json

# 3. Run download script
python scripts/download_imdb_dataset.py
```

**What the script does:**
- ✓ Checks if dataset already exists
- ✓ Downloads from Kaggle if API is configured
- ✓ Provides detailed manual instructions if API unavailable
- ✓ Validates file placement

**Option B: Manual Download**

The script also provides step-by-step manual instructions if you prefer not to set up the Kaggle API.

---

## Implementation Summary

### What's Automated:

| Feature | Status |
|---------|--------|
| MNIST download | ✅ Automatic |
| QM9 download | ✅ Automatic |
| Synthetic data generation | ✅ Automatic |
| IMDb download script | ✅ Created |
| IMDb manual instructions | ✅ Documented |
| Dependency installation | ✅ In SLURM scripts |

### Scripts Created:

1. ✅ `download_imdb_dataset.py` - Automated IMDb download
2. ✅ `DATASET_DOWNLOADS.md` - Complete download guide
3. ✅ `run_*.sh` - SLURM scripts with auto-installation

### Dependencies Handled:

```bash
# QM9 experiment
pip install torch-geometric torch-scatter torch-sparse -q

# Modular addition experiment
pip install einops -q

# All other dependencies already in environment
```

---

## Bottom Line

**Can datasets be downloaded programmatically?**

✅ **YES for 5/6 experiments** - Completely automatic

⚠️ **PARTIAL for 1/6 (IMDb)** - Automated IF Kaggle API is set up
   - Created helper script: `download_imdb_dataset.py`
   - Provides both automated and manual options
   - Only dataset requiring user intervention

**Total Automation Rate: 83%** (5/6 fully automated)

---

## Quick Command Summary

```bash
# Check what's already downloaded
ls mnist/grokking/MNIST/           # MNIST (auto)
ls qm9/grokking/qm9/               # QM9 (auto)
ls imdb/grokking/IMDB*.csv         # IMDb (manual/API)

# Download IMDb (if Kaggle API set up)
python scripts/download_imdb_dataset.py

# Or follow manual instructions
cat DATASET_DOWNLOADS.md

# Run all experiments
./scripts/run_all_experiments.sh
```

---

## Why IMDb Isn't Fully Automated

1. **Kaggle Terms of Service**: Requires user account
2. **API Authentication**: Needs personal API token
3. **Security**: Cannot store credentials in public code
4. **Solution**: Provided automated script that works **IF** user sets up API

This is the standard practice for Kaggle datasets - the download can be automated, but requires one-time user setup of API credentials.

All other datasets either:
- Auto-download via standard PyTorch functions ✅
- Are generated synthetically ✅
- Are included with required packages ✅

