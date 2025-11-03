# Running All Grokking Paper Replications

This guide shows how to run all 10 paper replications and generate plots.

## Quick Start

### Step 1: Run All Experiments
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications
./run_all_replications.sh
```

This will submit 10 SLURM jobs, one for each paper.

### Step 2: Monitor Progress
```bash
# Check all your jobs
squeue -u mabdel03

# Check specific job details
squeue -j JOB_ID

# View job output (while running or after completion)
tail -f 01_power_et_al_2022_openai_grok/logs/modular_addition_JOBID.out
```

### Step 3: Analyze Results
After jobs complete (or while they're running), generate plots:
```bash
python analyze_all_replications.py
```

This creates:
- Individual plots for each paper: `analysis_results/paper_XX_metrics.png`
- Comparison plot across all papers: `analysis_results/all_papers_comparison.png`
- Text summary: `analysis_results/analysis_summary.txt`

## What Each Paper Tests

1. **Power et al. (2022)** - Original grokking discovery on modular arithmetic
2. **Liu et al. (2022)** - Effective theory of grokking with toy models
3. **Nanda et al. (2023)** - Mechanistic interpretability of modular addition
4. **Wang et al. (2024)** - Grokking in knowledge graph reasoning
5. **Liu et al. (2022)** - Omnigrok: Grokking beyond algorithmic data (MNIST)
6. **Humayun et al. (2024)** - Deep networks always grok (MNIST MLP)
7. **Thilak et al. (2022)** - Slingshot mechanism in grokking
8. **Doshi et al. (2024)** - Grokking modular polynomials
9. **Levi et al. (2023)** - Grokking in linear estimators
10. **Minegishi et al. (2023)** - Lottery tickets accelerate grokking

## Expected Runtime

- **Short experiments** (1-4 hours): Papers 02, 09
- **Medium experiments** (4-12 hours): Papers 03, 07, 08, 10
- **Long experiments** (12-24 hours): Papers 01, 04, 05, 06

Total wall time (running in parallel): ~24 hours
Total compute time (if sequential): ~120 hours

## Troubleshooting

### Job Failed
Check the error log:
```bash
cat XX_paper_name/logs/*.err
```

### Out of Memory
Edit the SLURM script and increase `--mem`:
```bash
#SBATCH --mem=32G  # Increase from 16G
```

### Job Pending Too Long
Check available resources:
```bash
sinfo -o "%20P %5a %10l %6D %8G"
```

### Re-run a Single Paper
```bash
cd XX_paper_directory
sbatch run_*.sh
```

## Output Locations

Each paper saves results in its directory:
- **Logs**: `XX_paper_name/logs/`
- **Checkpoints**: `XX_paper_name/checkpoints/`
- **Plots** (from analyze script): `analysis_results/`

## Analysis Features

The analysis script automatically:
- ✅ Extracts train/test loss and accuracy from logs
- ✅ Generates loss curves (log scale) for each paper
- ✅ Generates accuracy curves for each paper
- ✅ Creates comparison plot across all 10 papers
- ✅ Detects grokking transitions (high train acc, delayed test acc improvement)
- ✅ Generates summary statistics

## Customization

### Run Specific Papers Only
Edit `run_all_replications.sh` and comment out papers you don't want:
```bash
# echo "[1/10] Submitting Power et al. (2022)..."
# cd 01_power_et_al_2022_openai_grok
# JOB_IDS[01]=$(sbatch --parsable run_modular_addition.sh)
# cd ..
```

### Adjust Training Parameters
Each paper's script has configurable parameters. Edit the `run_*.sh` file:
```bash
python train.py \
    --epochs=100000 \     # Adjust training length
    --lr=0.001 \          # Adjust learning rate
    --weight_decay=1.0    # Adjust regularization
```

### Change GPU Type
All scripts are configured for A100 GPUs. To use different GPUs:
```bash
#SBATCH --gres=gpu:RTXA5000:1  # Change A100 to RTXA5000
```

## Notes

- All scripts use the conda environment: `/om2/user/mabdel03/conda_envs/SLT_Proj_Env`
- All jobs run on partition: `use-everything`
- GPU requirement: At least 1 A100 GPU (80GB memory)
- Some experiments may take 24+ hours to show clear grokking

## Citation

If you use these replications, please cite the original papers. See `Prior_Works.tex` for complete citations.

