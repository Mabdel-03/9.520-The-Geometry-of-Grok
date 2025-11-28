# 📍 New_Explorations Directory Map

**Quick Navigation Guide** - Last Updated: November 25, 2024

---

## 🎯 Where to Start

```
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations

# For AGOP experiments (recommended):
cd agop_experiments/

# For standard experiments:
cd standard_experiments/
```

---

## 📁 Directory Guide

### [`agop_experiments/`](agop_experiments/) ⭐ RECOMMENDED

**Purpose:** Tractable input-gradient AGOP tracking for mechanistic analysis

**Key directories:**
- `core/` - AGOP implementation (agop_utils, onehot_datasets, onehot_models)
- `training_scripts/` - 4 training scripts with AGOP
- `tests/` - Verification tests (6/6 passing)
- `analysis/` - Visualization tools (9 plots per experiment)
- `slurm_scripts/` - Batch job submission
- `results/` - Experiment outputs

**Start here:** [`agop_experiments/README.md`](agop_experiments/README.md)

**Quick test:**
```bash
cd agop_experiments/
python tests/test_onehot_complete.py  # 6/6 tests pass
```

---

### [`standard_experiments/`](standard_experiments/)

**Purpose:** Original paper implementations without AGOP overhead

**Key directories:**
- `datasets/` - 4 dataset implementations
- `framework/` - Shared training code
- `slurm_scripts/` - Batch jobs
- `results/` - Outputs

**Start here:** [`standard_experiments/README.md`](standard_experiments/README.md)

---

### [`docs/`](docs/)

**Purpose:** Consolidated documentation

**Will contain:**
- Implementation history
- AGOP complete guide  
- Troubleshooting

---

## 🗺️ File Locations

### Need to run an AGOP experiment?
→ `agop_experiments/training_scripts/train_*_agop.py`

### Need to visualize AGOP results?
→ `agop_experiments/analysis/visualize_agop_metrics.py`

### Need to test if things work?
→ `agop_experiments/tests/test_onehot_complete.py`

### Need to submit batch jobs?
→ `agop_experiments/slurm_scripts/run_*_agop.sh`

### Need the AGOP implementation?
→ `agop_experiments/core/agop_utils.py`

### Need one-hot datasets?
→ `agop_experiments/core/onehot_datasets.py`

### Need model definitions?
→ `agop_experiments/core/onehot_models.py`

---

## 📊 What Changed

| Before | After |
|--------|-------|
| 41+ .md files | 6 essential .md files |
| Mixed organization | Clear separation |
| Files scattered | Organized by type |
| Relative paths | Absolute paths |
| Unclear entry point | README.md navigation |

---

## ✅ Verification

All systems operational:
- ✓ 6/6 component tests passing
- ✓ All imports resolve
- ✓ AGOP computation works
- ✓ Training pipelines functional

---

**Use this file for quick navigation!**
