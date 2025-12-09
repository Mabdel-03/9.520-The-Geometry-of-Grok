# SLURM Scripts

This directory contains SLURM job submission scripts for running AGOP-tracking experiments on the OpenMind cluster. Scripts are organized by dataset and experiment scope (single runs, weight decay sweeps, full factorial sweeps).

---

## Directory Structure

```
slurm_scripts/
├── logs/                           # Job output logs
├── run_nanda_agop.sh              # Nanda single sweep
├── run_nanda_full_sweep.sh        # Nanda full factorial (2 arch × 3 opt × 4 wd)
├── run_nanda_wd_sweep.sh          # Nanda weight decay sweep
├── run_softmax_agop.sh            # Softmax single sweep
├── run_softmax_full_sweep.sh      # Softmax full factorial
├── run_softmax_wd_sweep.sh        # Softmax weight decay sweep
├── run_mnist_agop.sh              # MNIST single sweep
├── run_mnist_full_sweep.sh        # MNIST full factorial
├── run_mnist_wd_sweep.sh          # MNIST weight decay sweep
├── run_composition_agop.sh        # Composition single sweep
├── run_composition_full_sweep.sh  # Composition full factorial
├── run_all_agop.sh               # Submit all datasets
├── submit_all_experiments.sh      # Master submission script
├── submit_all_lazy_rich.sh        # Submit with lazy-rich tracking
├── submit_wd_sweep.sh             # Submit weight decay sweeps
├── resubmit_failed.sh            # Resubmit failed jobs
└── resubmit_transformer_jobs.sh   # Resubmit transformer-specific jobs
```

---

## Resource Requirements

### Default SLURM Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Time Limit | 48:00:00 | 48 hours per job |
| Memory | 96 GB | RAM allocation |
| CPUs | 4 | CPU cores per task |
| GPU | 1 × GEFORCE RTX 2080 | GPU allocation |
| Partition | normal | Cluster partition |

### Per-Dataset Estimates

| Dataset | ~Time/Experiment | Memory Usage | GPU Memory |
|---------|------------------|--------------|------------|
| Nanda | 4–8 hours | ~20 GB | ~4 GB |
| Softmax | 6–12 hours | ~25 GB | ~6 GB |
| MNIST | 8–16 hours | ~30 GB | ~4 GB |
| Composition | 12–24 hours | ~20 GB | ~4 GB |

---

## Script Categories

### 1. Full Sweep Scripts (`*_full_sweep.sh`)

Run complete factorial experiments across architectures, optimizers, and weight decays.

**Example: `run_nanda_full_sweep.sh`**

```bash
#SBATCH --array=0-23  # 2 arch × 3 opt × 4 wd = 24 jobs

ARCHITECTURES=("mlp" "transformer")
OPTIMIZERS=("adamw" "muon" "sgd")
WEIGHT_DECAYS=(0.1 1.0 5.0 10.0)
```

| Script | Array Size | Architectures | Optimizers | Weight Decays |
|--------|------------|---------------|------------|---------------|
| `run_nanda_full_sweep.sh` | 0-23 | mlp, transformer | adamw, muon, sgd | 0.1, 1.0, 5.0, 10.0 |
| `run_softmax_full_sweep.sh` | 0-23 | mlp, transformer | adamw, muon, sgd | 0.01, 0.1, 0.5, 1.0 |
| `run_mnist_full_sweep.sh` | 0-11 | mlp | adamw, muon, sgd | 0.01, 0.1, 0.5, 1.0 |
| `run_composition_full_sweep.sh` | 0-11 | mlp | adamw, muon, sgd | 0.01, 0.1, 0.5, 1.0 |

### 2. Single Sweep Scripts (`*_agop.sh`)

Run optimizer sweep with fixed weight decay configurations.

### 3. Weight Decay Sweep Scripts (`*_wd_sweep.sh`)

Focused weight decay exploration for specific optimizer/architecture combinations.

### 4. Utility Scripts

| Script | Purpose |
|--------|---------|
| `run_all_agop.sh` | Submit all dataset sweeps sequentially |
| `submit_all_experiments.sh` | Master script for full experiment matrix |
| `submit_all_lazy_rich.sh` | Submit experiments with lazy-rich tracking enabled |
| `resubmit_failed.sh` | Identify and resubmit failed jobs |
| `resubmit_transformer_jobs.sh` | Resubmit transformer-specific experiments |

---

## Usage

### Submit Single Dataset Sweep

```bash
cd slurm_scripts/
sbatch run_nanda_full_sweep.sh
```

### Submit All Experiments

```bash
# Submit all datasets (72 total jobs)
./submit_all_experiments.sh

# Or individually
sbatch run_nanda_full_sweep.sh    # 24 jobs
sbatch run_softmax_full_sweep.sh  # 24 jobs
sbatch run_mnist_full_sweep.sh    # 12 jobs
sbatch run_composition_full_sweep.sh  # 12 jobs
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View running jobs
squeue -u $USER -t RUNNING

# Check specific job array
squeue -u $USER -j <job_id>
```

### Check Logs

```bash
# View stdout
tail -f logs/nanda_agop_<job_id>_<array_id>.out

# View stderr
tail -f logs/nanda_agop_<job_id>_<array_id>.err

# Check for failures
grep -l "Error" logs/*.err
```

### Resubmit Failed Jobs

```bash
# Identify failed jobs
./resubmit_failed.sh

# Manually resubmit specific array indices
sbatch --array=5,12,18 run_nanda_full_sweep.sh
```

---

## Script Anatomy

### Full Sweep Script Structure

```bash
#!/bin/bash
#SBATCH --job-name=nanda_agop
#SBATCH --output=logs/nanda_agop_%A_%a.out
#SBATCH --error=logs/nanda_agop_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GEFORCERTX2080:1
#SBATCH --partition=normal

# Environment setup
CONDA_ENV=/om/scratch/Mon/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

mkdir -p logs

# Parameter arrays
ARCHITECTURES=("mlp" "transformer")
OPTIMIZERS=("adamw" "muon" "sgd")
WEIGHT_DECAYS=(0.1 1.0 5.0 10.0)

# Index calculation: arch × (opt × wd)
ARCH_IDX=$((SLURM_ARRAY_TASK_ID / 12))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 12))
OPT_IDX=$((REMAINDER / 4))
WD_IDX=$((REMAINDER % 4))

ARCHITECTURE=${ARCHITECTURES[$ARCH_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
WEIGHT_DECAY=${WEIGHT_DECAYS[$WD_IDX]}

# Run training
$CONDA_ENV/bin/python ../training_scripts/train_nanda_agop.py \
    --architecture $ARCHITECTURE \
    --optimizer $OPTIMIZER \
    --weight_decay $WEIGHT_DECAY \
    --lr 0.001 \
    --p 113 \
    --train_fraction 0.3 \
    --n_epochs 40000 \
    --agop_freq 100 \
    --agop_top_k 20 \
    --ntk_subsample 200 \
    --log_freq 100 \
    --device cuda \
    --seed 42 \
    --save_dir ../results/nanda \
    --experiment_name nanda_${ARCHITECTURE}_${OPTIMIZER}_wd${WEIGHT_DECAY}_seed42
```

### Array Index Mapping

For a full sweep with 2 architectures × 3 optimizers × 4 weight decays = 24 jobs:

| Array ID | Architecture | Optimizer | Weight Decay |
|----------|--------------|-----------|--------------|
| 0 | mlp | adamw | 0.1 |
| 1 | mlp | adamw | 1.0 |
| 2 | mlp | adamw | 5.0 |
| 3 | mlp | adamw | 10.0 |
| 4 | mlp | muon | 0.1 |
| ... | ... | ... | ... |
| 12 | transformer | adamw | 0.1 |
| ... | ... | ... | ... |
| 23 | transformer | sgd | 10.0 |

---

## Environment Configuration

### Conda Environment

Scripts expect a conda environment at:
```
/om/scratch/Mon/mabdel03/9.520/conda_envs/grok_exp
```

### Required Packages

```
torch >= 1.9.0
torchvision
numpy
h5py
tqdm
pyyaml
```

### Environment Activation

```bash
CONDA_ENV=/om/scratch/Mon/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH
```

---

## Output Organization

Results are saved to:
```
../results/{dataset}/{experiment_name}/
├── config.json
├── training_history.json
├── agop_metrics.h5
└── lazy_rich_metrics.h5
```

**Naming convention**: `{dataset}_{architecture}_{optimizer}_wd{weight_decay}_seed{seed}`

Example: `nanda_transformer_adamw_wd1.0_seed42`

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Check conda environment path |
| `CUDA out of memory` | Reduce batch size or NTK subsample |
| Job timeout | Increase `--time` limit |
| Missing GPU | Verify `--gres=gpu:GEFORCERTX2080:1` |

### Debugging Commands

```bash
# Check GPU availability
srun --gres=gpu:1 --pty nvidia-smi

# Test single job interactively
srun --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash
source /om/scratch/Mon/mabdel03/9.520/conda_envs/grok_exp/bin/activate
python ../training_scripts/train_nanda_agop.py --n_epochs 100 --device cuda

# Check job efficiency
sacct -j <job_id> --format=JobID,Elapsed,MaxRSS,MaxVMSize,State
```

### Log Analysis

```bash
# Find errors
grep -r "Error\|Exception\|Traceback" logs/

# Check job completion
grep -l "Job completed" logs/*.out | wc -l

# Find jobs that didn't complete
for f in logs/*.out; do
    if ! grep -q "Job completed" "$f"; then
        echo "$f"
    fi
done
```

---

## Experiment Status

### Current Completion

| Dataset | Total Jobs | Completed | With AGOP | With Lazy-Rich |
|---------|------------|-----------|-----------|----------------|
| Nanda | 24 | 24 | 24 | 24 |
| Softmax | 24 | 24 | 24 | 24 |
| MNIST | 12 | 12 | 12 | 12 |
| Composition | 12 | 12 | 12 | 0 |
| **Total** | **72** | **72** | **72** | **60** |

---

## References

- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [OpenMind User Guide](https://github.mit.edu/MGHPCC/OpenMind/wiki)

---

*Last Updated: December 2024*







