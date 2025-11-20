# Replications Directory Structure

**Last Updated**: November 19, 2025  
**Status**: Reorganized and Standardized

---

## Standardized Structure

All paper directories now follow this consistent structure:

```
XX_paper_name/
├── README.md                    # Comprehensive paper information
├── scripts/                     # All executable files
│   ├── train.py                 # Main training script
│   ├── model.py                 # Model definitions (if separate)
│   ├── run_*.sh                 # SLURM submission scripts
│   ├── extract_*.py             # Data extraction scripts
│   └── ...                      # Other analysis scripts
├── results/                     # All training outputs
│   ├── logs/
│   │   ├── training_history.json  # Standardized metrics
│   │   └── *.out, *.err          # SLURM logs
│   └── checkpoints/             # Model checkpoints
├── data/                        # Dataset files (if applicable)
├── config/                      # Configuration files (if applicable)
└── [original subdirs]           # Preserved from original repos
```

---

## Papers with Confirmed Data

| Paper | training_history.json | README.md | scripts/ | results/ |
|-------|----------------------|-----------|----------|----------|
| 01 | ✅ results/logs/ | ✅ Merged | ✅ | ✅ |
| 02 | ✅ results/experiment_1_toy_model/ | ✅ Single | ✅ | ✅ |
| 03 | ✅ results/logs/ | ✅ Single | ✅ | ✅ |
| 04 | ❌ Not run | ✅ Merged | ✅ | ✅ |
| 05 | ✅ results/logs/ | ✅ Merged | ✅ | ✅ |
| 06 | ✅ results/logs/ | ✅ Single | ✅ | ✅ |
| 07 | ✅ results/logs/ | ✅ Single | ✅ | ✅ |
| 08 | ✅ results/logs/ | ✅ Single | ✅ | ✅ |
| 09 | ✅ results/logs/ | ✅ Single | ✅ | ✅ |
| 10 | ❌ Not run | ✅ Merged | ✅ | ✅ |

---

## Quick Access Paths

### Training Data
```bash
# All papers (except Paper 02):
XX_paper_name/results/logs/training_history.json

# Paper 02:
02_liu_et_al_2022_effective_theory/results/experiment_1_toy_model/training_history.json
```

### Scripts
```bash
# All training scripts:
XX_paper_name/scripts/train.py

# SLURM submission:
XX_paper_name/scripts/run_*.sh
```

### Results
```bash
# Logs:
XX_paper_name/results/logs/

# Checkpoints:
XX_paper_name/results/checkpoints/
```

---

## Benefits of New Structure

1. **Consistency**: All papers have same organization
2. **Clarity**: Scripts separated from results
3. **Single README**: One comprehensive file per paper
4. **Easy Navigation**: Know exactly where to find files
5. **Preserved Structure**: Original repo subdirectories maintained

---

## File Counts

- Total papers: 10
- Papers with training_history.json: 8
- Papers with standardized structure: 10
- Papers with single README: 10

---

## Navigation Examples

### To find training scripts:
```bash
ls */scripts/*.py
```

### To find all results:
```bash
ls */results/logs/training_history.json
```

### To run an experiment:
```bash
cd XX_paper_name
sbatch scripts/run_*.sh
```

### To check results:
```bash
cd XX_paper_name/results/logs
python ../../../plot_paperXX_grokking.py
```

---

**Repository is now clean, organized, and intuitively structured!**

