# Dataset Downloads - Quick Reference

## Summary

**Can datasets be downloaded programmatically?**

**YES for 5/6 experiments** - Fully automatic  
**PARTIAL for 1/6 (IMDb)** - Requires Kaggle API setup

---

## Quick Commands

### Check What is Downloaded
```bash
# MNIST (should auto-download)
ls mnist/grokking/MNIST/

# QM9 (should auto-download)  
ls qm9/grokking/qm9/

# IMDb (needs manual download or Kaggle API)
ls imdb/grokking/IMDB\ Dataset.csv
```

### Download IMDb Dataset

**Option 1: Automatic (if Kaggle API set up)**
```bash
python scripts/download_imdb_dataset.py
```

**Option 2: Manual**
1. Visit: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Download `IMDB Dataset.csv`
3. Place in: `imdb/grokking/`

---

## What is Automatic

1. **MNIST** - Downloads automatically via PyTorch
2. **QM9** - Downloads automatically via PyTorch Geometric  
3. **Teacher-Student** - Synthetic data (no download)
4. **Modular Addition** - Synthetic data (no download)
5. **MNIST Repr** - Same as MNIST

## What Needs Setup

**IMDb** - One-time Kaggle API setup:
```bash
# 1. Install
pip install kaggle

# 2. Get API token
# Visit: https://www.kaggle.com/settings
# Click "Create New API Token"
# Save to ~/.kaggle/kaggle.json

# 3. Download
python scripts/download_imdb_dataset.py

# 4. Run experiment
sbatch scripts/run_imdb.sh
```

---

## Full Details

See `DATASET_DOWNLOADS.md` for complete guide

See `PAPER05_DATASET_AUTOMATION.md` for technical analysis
