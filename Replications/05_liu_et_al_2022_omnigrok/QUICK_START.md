# Paper 05: Omnigrok - Quick Start Guide

## Monitor Running Experiments

```bash
# Check job status
squeue -u mabdel03 | grep paper05

# Watch live progress
tail -f results/logs/mnist_corrected_*.out
tail -f results/logs/mnist_repr_*.out

# Check completed logs
ls -lth results/logs/*.out | head
```

## When Experiments Complete

```bash
# Generate all plots
python plot_all_results.py

# View results
ls -lh results/*.png
ls -lh results/logs/*.json

# Check for grokking
cat results/logs/training_history.json | grep -A5 "train_acc.*1.0"
```

## Run Missing Experiment (IMDb)

```bash
# After downloading IMDB Dataset.csv to imdb/grokking/
sbatch scripts/run_imdb.sh
```

## Resubmit if Needed

```bash
cd scripts
sbatch run_mnist_corrected.sh    # ~2-4 hours
sbatch run_qm9.sh                 # ~8-12 hours  
sbatch run_teacher_student.sh    # ~4-6 hours
sbatch run_modular_addition.sh   # ~6-8 hours
sbatch run_mnist_repr.sh          # ~2-4 hours
```

## Key Files

- `PAPER05_IMPLEMENTATION_COMPLETE.md` - Full summary
- `PAPER05_VERIFICATION_REPORT.md` - Detailed verification
- `EXPERIMENTS_STATUS.md` - Live status tracking
- `plot_all_results.py` - Automated visualization

## Expected Timeline

- **MNIST**: 2-4 hours → Smooth grokking
- **Teacher-Student**: Complete (needs analysis)
- **MNIST Repr**: 2-4 hours → Representation dynamics
- **QM9**: 8-12 hours → Clear grokking
- **Modular Addition**: 6-8 hours → Sharp transitions

## Success Criteria

Each experiment should show:
1. ✓ High train accuracy (memorization)
2. ✓ Delayed test accuracy improvement (generalization)
3. ✓ Final small gap (~10-15%)

All specifications match paper exactly!

