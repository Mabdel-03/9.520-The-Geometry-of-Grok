# Paper 05: Omnigrok - Dataset Download Guide

This guide explains which datasets are downloaded automatically and which require manual intervention.

---

## ✅ **Automatic Downloads** (No Action Needed)

These datasets download automatically when you run the experiments:

### 1. **MNIST** (Experiments 1 & 6)
- **Status**: ✅ **Automatic**
- **Method**: PyTorch's `torchvision.datasets.MNIST`
- **Code**: `download=True` parameter handles everything
- **Size**: ~11 MB
- **Location**: Downloads to working directory

### 2. **QM9 Molecules** (Experiment 3)
- **Status**: ✅ **Automatic** 
- **Method**: PyTorch Geometric's `torch_geometric.datasets.QM9`
- **Code**: Automatically downloads on first use
- **Size**: Variable
- **Note**: Requires `torch-geometric` package (installed via `run_qm9.sh`)

### 3. **Teacher-Student** (Experiment 4)
- **Status**: ✅ **Synthetic Data** (No download needed)
- **Method**: Generated programmatically in Python script
- **Details**: Creates random Gaussian data

### 4. **Modular Addition** (Experiment 5)
- **Status**: ✅ **Synthetic Data** (No download needed)
- **Method**: Generated using `torch.meshgrid`
- **Details**: Creates modular arithmetic dataset (p=113)

---

## ⚠️ **Manual Download Required**

### IMDb Sentiment Dataset (Experiment 2)

**Current Status**: ❌ **Requires Manual Download**

#### Option A: Kaggle API (Recommended)

**Prerequisites:**
1. Install Kaggle package:
   ```bash
   pip install kaggle
   ```

2. Set up API credentials:
   - Visit: https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Save `kaggle.json` to `~/.kaggle/kaggle.json`

**Download Command:**
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/05_liu_et_al_2022_omnigrok/scripts
python download_imdb_dataset.py
```

The script will:
- Check if dataset already exists
- Download from Kaggle if credentials are set up
- Provide manual download instructions if needed

#### Option B: Manual Download

1. **Visit Kaggle**:
   ```
   https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
   ```

2. **Download the file**:
   - Click "Download" button
   - Get `IMDB Dataset.csv` (~65 MB)

3. **Place the file**:
   ```bash
   mv ~/Downloads/"IMDB Dataset.csv" \
      /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/05_liu_et_al_2022_omnigrok/imdb/grokking/
   ```

4. **Verify placement**:
   ```bash
   ls -lh /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/05_liu_et_al_2022_omnigrok/imdb/grokking/IMDB\ Dataset.csv
   ```

5. **Run the experiment**:
   ```bash
   cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/05_liu_et_al_2022_omnigrok/scripts
   sbatch run_imdb.sh
   ```

---

## Summary Table

| Experiment | Dataset | Download Method | Status |
|------------|---------|-----------------|--------|
| MNIST (1) | MNIST | Automatic (torchvision) | ✅ |
| IMDb (2) | IMDb Reviews | **Manual/Kaggle API** | ⚠️ |
| QM9 (3) | Molecules | Automatic (torch_geometric) | ✅ |
| Teacher-Student (4) | Synthetic | Generated in code | ✅ |
| Modular Addition (5) | Synthetic | Generated in code | ✅ |
| MNIST Repr (6) | MNIST | Automatic (torchvision) | ✅ |

---

## Troubleshooting

### Issue: Kaggle API Authentication Fails

**Solution:**
```bash
# Check if credentials file exists
ls -la ~/.kaggle/kaggle.json

# Verify permissions (should be 600)
chmod 600 ~/.kaggle/kaggle.json

# Test API
kaggle datasets list
```

### Issue: torch-geometric Installation Fails

**Solution:**
```bash
# Install with specific CUDA version
pip install torch-scatter torch-sparse torch-geometric -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
```
(Adjust CUDA version as needed for your system)

### Issue: Dataset Already Downloaded but Not Found

**Check these locations:**
```bash
# MNIST
ls mnist/grokking/MNIST/

# QM9
ls qm9/grokking/qm9/

# IMDb
ls imdb/grokking/IMDB\ Dataset.csv
```

---

## Quick Start After Download

Once all datasets are available:

```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/05_liu_et_al_2022_omnigrok/scripts

# Run all experiments
./run_all_experiments.sh

# Or run individually
sbatch run_mnist_corrected.sh
sbatch run_imdb.sh              # Only after IMDb download
sbatch run_qm9.sh
sbatch run_teacher_student.sh
sbatch run_modular_addition.sh
sbatch run_mnist_repr.sh
```

---

## Dataset Sizes

| Dataset | Approximate Size |
|---------|------------------|
| MNIST | ~11 MB |
| IMDb | ~65 MB |
| QM9 | Variable (molecular data) |
| Synthetic | Generated on-the-fly |

---

## Notes

1. **MNIST** downloads automatically the first time you run the experiment
2. **QM9** may take longer to download depending on network speed
3. **IMDb** is the only dataset requiring manual intervention
4. All other data is **generated synthetically** during training

---

## Help

For issues with dataset downloads:

1. Check network connection
2. Verify disk space (at least 1 GB free)
3. Check file permissions in target directories
4. Consult experiment logs in `results/logs/`

For Kaggle API issues:
- Documentation: https://github.com/Kaggle/kaggle-api
- Credentials setup: https://www.kaggle.com/docs/api

