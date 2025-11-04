# Grokking Experiments - Current Status

**Date**: November 3, 2025  
**Location**: OpenMind HPC

## Running Experiments (6/10) ✅

| Paper | Title | Job ID | Status | Notes |
|-------|-------|---------|--------|-------|
| 01 | Power et al. (2022) - OpenAI Grok | 44183354 | ✅ RUNNING | Modular addition (mod 97) |
| 03 | Nanda et al. (2023) - Progress Measures | 44183356 | ✅ RUNNING | Mechanistic interpretability |
| 06 | Humayun et al. (2024) - Deep Networks | 44183359 | ✅ RUNNING | MNIST MLP grokking |
| 07 | Thilak et al. (2022) - Slingshot | 44183360 | ✅ RUNNING | Slingshot mechanism |
| 08 | Doshi et al. (2024) - Modular Polynomials | 44183361 | ✅ RUNNING | Polynomial grokking |
| 09 | Levi et al. (2023) - Linear Estimators | 44183362 | ✅ COMPLETED | 100k epochs finished! |

## Experiments Needing Fixes (4/10) ⚠️

### Paper 02: Liu et al. (2022) - Effective Theory
**Issue**: Missing `sacred` module  
**Fix**: 
```bash
pip install sacred
```

### Paper 04: Wang et al. (2024) - Knowledge Graphs  
**Issue**: Missing `train.py` file in expected location  
**Fix**: Check repository structure, may need to adapt script

### Paper 05: Liu et al. (2022) - Omnigrok
**Issue**: Jupyter notebook conversion failed  
**Fix**: Need to properly convert `.ipynb` to `.py` or run notebook directly

### Paper 10: Minegishi et al. (2023) - Lottery Tickets
**Issue**: Missing `main.py` file  
**Fix**: Check repository structure, may need different entry point

## Successful Completion Example

**Paper 09** (Levi et al. - Linear Estimators) completed successfully:
- **Epochs**: 100,000
- **Final Train Loss**: 0.000006
- **Final Train Accuracy**: 100.0% (perfect memorization)
- **Final Test Loss**: 0.214455  
- **Final Test Accuracy**: 5.72% (poor generalization)
- **Grokking Status**: Shows classic memorization phase
- **Output**: Saved to `logs/training_history.json`

This demonstrates the **grokking phenomenon**: The model perfectly memorizes the training data (100% accuracy) but has not yet generalized to test data (5.7% accuracy). With continued training and proper weight decay, test accuracy should eventually jump up.

## Monitoring Commands

### Check running jobs:
```bash
squeue -u mabdel03
```

### View live output:
```bash
tail -f 09_levi_et_al_2023_linear_estimators/linear_1layer_44183362.out
```

### Check all job statuses:
```bash
sacct -u mabdel03 --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed
```

## Generate Plots

Once experiments complete (or while running):
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications
python analyze_all_replications.py
```

This will:
- Extract train/test metrics from all log files
- Generate individual loss/accuracy plots for each paper
- Create comparison plot across all papers
- Generate summary statistics
- Output to `analysis_results/` directory

## Expected Training Times

| Paper | Parameters | Epochs | Estimated Time |
|-------|-----------|--------|----------------|
| 09 | 1K | 100K | ~2 hours ✅ DONE |
| 03 | 100K | 40K | ~12 hours |
| 07 | 150K | 50K | ~18 hours |
| 08 | 50K | 50K | ~10 hours |
| 01 | 200K | 100K | ~20 hours |
| 06 | 160K | 100K | ~24 hours |

## Next Steps

1. **For running experiments**: Monitor progress periodically
2. **For failed experiments**: Apply fixes and resubmit:
   ```bash
   cd XX_paper_directory
   sbatch run_*.sh
   ```
3. **After completion**: Run analysis script to generate plots
4. **Documentation**: See `RUN_ALL_GUIDE.md` for detailed instructions

## Training Data Format

All experiments log metrics in format:
```
Epoch XXXXX | Train Loss: X.XXXXXX | Train Acc: X.XXXX | Test Loss: X.XXXXXX | Test Acc: X.XXXX
```

This format is automatically parsed by `analyze_all_replications.py` to generate plots.

## Git Repository

All changes pushed to: https://github.com/Mabdel-03/9.520-The-Geometry-of-Grok

Latest commit includes:
- Updated SLURM scripts for OpenMind HPC
- Master script to run all replications
- Analysis script for automatic plot generation  
- Fixed output paths to avoid SLURM errors

