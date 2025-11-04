# How to Monitor Your Grokking Experiments

## Quick Status Check

### See all your running jobs:
```bash
squeue -u mabdel03
```

### See only grokking experiments:
```bash
squeue -u mabdel03 | grep grok
```

### Detailed view with runtime:
```bash
squeue -u mabdel03 --format="%.10i %.30j %.10P %.8T %.15M %.6D %12R"
```

### Check which papers have data:
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications
./check_all_status.sh
```

---

## Monitor Real-Time Training Progress

### For Papers with `training_history.json` (Papers 03, 07, 08, 09)

**Watch the training output live:**
```bash
# Paper 03 (Nanda)
tail -f 03_nanda_et_al_2023_progress_measures/modular_addition_*.out

# Paper 07 (Thilak) - Currently running!
tail -f 07_thilak_et_al_2022_slingshot/slingshot_*.out

# Paper 08 (Doshi)
tail -f 08_doshi_et_al_2024_modular_polynomials/modular_addition_*.out

# Paper 09 (Levi)
tail -f 09_levi_et_al_2023_linear_estimators/linear_1layer_*.out
```

**See latest training metrics:**
```bash
# Get last 20 lines
tail -20 07_thilak_et_al_2022_slingshot/slingshot_*.out

# Get last 50 lines to see progress
tail -50 07_thilak_et_al_2022_slingshot/slingshot_*.out
```

**Check the JSON data directly:**
```bash
# See how many epochs have been logged
wc -l 07_thilak_et_al_2022_slingshot/logs/training_history.json

# Quick Python check of current progress
cd 07_thilak_et_al_2022_slingshot
python << 'EOF'
import json
with open('logs/training_history.json') as f:
    data = json.load(f)
print(f"Epochs: {data['epoch'][0]} to {data['epoch'][-1]}")
print(f"Train acc: {data['train_acc'][-1]:.2%}")
print(f"Test acc: {data['test_acc'][-1]:.2%}")
print(f"Checkpoints: {len(data['epoch'])}")
EOF
```

---

## Monitor Specific Papers

### Paper 05 (Omnigrok - MNIST) - Currently Running

```bash
# Watch training
tail -f 05_liu_et_al_2022_omnigrok/mnist_*.out

# The output shows: "L: X.Xe+02|X.Xe+02. A: XX.X%|XX.X%"
# This means: Loss: train|test. Accuracy: train%|test%

# Check error log for epoch number
tail 05_liu_et_al_2022_omnigrok/mnist_*.err | grep -E "it/s|epoch"
```

### Paper 06 (Humayun - Deep Networks) - Currently Running

```bash
# Watch training  
tail -f 06_humayun_et_al_2024_deep_networks/mnist_mlp_*.out

# Check if it's generating checkpoints
ls -lth 06_humayun_et_al_2024_deep_networks/checkpoints/ 2>/dev/null || echo "No checkpoints yet"

# Check logs
ls -lth 06_humayun_et_al_2024_deep_networks/logs/ 2>/dev/null
```

### Paper 07 (Thilak - Slingshot) - Currently Running 

```bash
# Watch training
tail -f 07_thilak_et_al_2022_slingshot/slingshot_*.out

# Check current epoch
cd 07_thilak_et_al_2022_slingshot
python << 'EOF'
import json
with open('logs/training_history.json') as f:
    d = json.load(f)
current = d['epoch'][-1]
total = 300000
progress = (current / total) * 100
print(f"Progress: Epoch {current:,}/{total:,} ({progress:.1f}%)")
print(f"Train acc: {d['train_acc'][-1]:.2%}")
print(f"Test acc: {d['test_acc'][-1]:.2%}")
EOF
```

---

## Check Job Status History

### See all completed jobs from today:
```bash
sacct -u mabdel03 --starttime=today --format=JobID,JobName,State,Elapsed,End | grep grok
```

### Check specific job details:
```bash
# Replace JOB_ID with actual job number
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,Start,End

# Example for Paper 07 (find job ID first with squeue)
sacct -j 44183393 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
```

### Find job output files:
```bash
# List all output files sorted by time
ls -lth */*.out */*.err | head -20

# Find files for specific job ID
find . -name "*44183393*"
```

---

## Quick Progress Checks

### One-liner to see all training data status:
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications

for i in {03..09}; do
    json="$(ls ${i}_*/logs/training_history.json 2>/dev/null | head -1)"
    if [ -n "$json" ]; then
        python3 << EOF
import json
with open("$json") as f:
    d = json.load(f)
print(f"Paper $i: {len(d['epoch'])} epochs | Test acc: {d['test_acc'][-1]:.2%}")
EOF
    else
        echo "Paper $i: No data yet"
    fi
done
```

### Check if grokking is happening:
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications

python3 << 'EOF'
import json
import numpy as np
from pathlib import Path

for paper_num in ["03", "07", "08", "09"]:
    json_files = list(Path(".").glob(f"{paper_num}_*/logs/training_history.json"))
    if not json_files:
        continue
    
    with open(json_files[0]) as f:
        d = json.load(f)
    
    test_acc = np.array(d['test_acc'])
    train_acc = np.array(d['train_acc'])
    epochs = np.array(d['epoch'])
    
    # Check for grokking
    grokked = "✅ GROKKED" if test_acc[-1] > 0.90 else "⏳ Training"
    memorized = "✓" if train_acc[-1] > 0.95 else "✗"
    
    # Find largest jump
    diffs = np.diff(test_acc)
    if len(diffs) > 0:
        max_jump = diffs.max()
        jump_epoch = epochs[diffs.argmax()] if max_jump > 0.05 else "none"
    else:
        max_jump, jump_epoch = 0, "none"
    
    print(f"Paper {paper_num}: {grokked} | Mem:{memorized} | "
          f"Test:{test_acc[-1]:.1%} | Max jump:{max_jump:.1%} @ epoch {jump_epoch}")
EOF
```

---

## Monitor GPU Usage

### Check GPU utilization:
```bash
# See which GPUs your jobs are using
squeue -u mabdel03 --format="%.10i %.30j %12R %10P"

# If you have access to the compute node:
ssh node108  # Replace with your node
nvidia-smi
```

---

## Estimate Time Remaining

### For Paper 07 (Slingshot):
```bash
cd 07_thilak_et_al_2022_slingshot
python << 'EOF'
import json
import time

with open('logs/training_history.json') as f:
    d = json.load(f)

current_epoch = d['epoch'][-1]
total_epochs = 300000
checkpoints = len(d['epoch'])

# Assuming running for 14 hours and at checkpoint X
hours_running = 14
epochs_per_hour = current_epoch / hours_running
hours_remaining = (total_epochs - current_epoch) / epochs_per_hour

print(f"Current: Epoch {current_epoch:,}/{total_epochs:,}")
print(f"Speed: ~{epochs_per_hour:,.0f} epochs/hour")
print(f"Time remaining: ~{hours_remaining:.1f} hours")
print(f"Est. completion: {hours_remaining + hours_running:.1f} total hours")
EOF
```

### For Paper 05 (Omnigrok):
```bash
# Check progress from the error log (shows iteration number)
tail 05_liu_et_al_2022_omnigrok/mnist_*.err | grep -E "[0-9]+/100000"

# Each iteration takes ~2 seconds
# Total time estimate: 100,000 * 2s = 200,000s = 55 hours
```

---

## Generate Updated Plots

### While experiments are running:
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications

# Activate environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Generate plots for all available data
python plot_results.py

# Generate detailed grokking analysis for Paper 03
python plot_grokking_detail.py
```

---

## Troubleshooting

### Job disappeared from queue but didn't finish:

```bash
# Check recent job history
sacct -u mabdel03 --starttime=today | grep grok

# Check why it failed
sacct -j JOB_ID --format=JobID,State,ExitCode,Reason
```

### No output files appearing:

```bash
# Make sure you're in the right directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications

# List all recent files
find . -name "*.out" -o -name "*.err" | xargs ls -lt | head -20
```

### Training seems stuck:

```bash
# Check if file is still being written to
watch -n 5 "ls -lh 07_thilak_et_al_2022_slingshot/slingshot_*.out"

# Check if JSON is being updated
watch -n 10 "tail -2 07_thilak_et_al_2022_slingshot/logs/training_history.json"
```

---

## Automated Monitoring Loop

Create a monitoring script that runs periodically:

```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications

# Run every 30 minutes
while true; do
    clear
    echo "=== Status at $(date) ==="
    ./check_all_status.sh
    echo ""
    echo "Next update in 30 minutes... (Ctrl+C to stop)"
    sleep 1800
done
```

Or use the provided script:

```bash
watch -n 300 ./check_all_status.sh  # Updates every 5 minutes
```

---

## Current Jobs to Monitor

Based on the latest status, here are the specific commands for your running jobs:

```bash
# Paper 05 (Omnigrok) - Job 44183398 - RUNNING
tail -f 05_liu_et_al_2022_omnigrok/mnist_*.out

# Paper 06 (Humayun) - Jobs 44183359, 44183392 - RUNNING  
tail -f 06_humayun_et_al_2024_deep_networks/mnist_mlp_*.out

# Paper 07 (Thilak/Slingshot) - Job 44183393 - RUNNING
tail -f 07_thilak_et_al_2022_slingshot/slingshot_*.out

# Quick check all at once
tail 05_*/mnist_*.err 06_*/mnist_*.out 07_*/slingshot_*.out | grep -E "Epoch|it/s|Loss|Acc"
```

---

## Best Practice: Monitor in tmux/screen

```bash
# Start tmux session
tmux new -s grokking_monitor

# Split into panes (Ctrl+b then ")
# In each pane, monitor a different experiment:

# Pane 1: Overall status
watch -n 60 'cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications && ./check_all_status.sh'

# Pane 2: Paper 05 progress
tail -f 05_liu_et_al_2022_omnigrok/mnist_*.err

# Pane 3: Paper 07 progress  
tail -f 07_thilak_et_al_2022_slingshot/slingshot_*.out

# Pane 4: Job queue
watch -n 30 'squeue -u mabdel03'

# Detach: Ctrl+b then d
# Reattach later: tmux attach -t grokking_monitor
```

---

## Email Notifications (Optional)

Add to SLURM scripts:
```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@example.com
```

---

## Summary: Your Current Active Experiments

| Paper | Job ID | Status | Command to Monitor |
|-------|--------|--------|-------------------|
| 03 | Done | ✅ Complete | See plots in `analysis_results/` |
| 05 | 44183398 | 🏃 Running | `tail -f 05_*/mnist_*.err` |
| 06 | 44183359 | 🏃 Running | `tail -f 06_*/mnist_mlp_*.out` |
| 06 | 44183392 | 🏃 Running | Same as above |
| 07 | 44183393 | 🏃 Running | `tail -f 07_*/slingshot_*.out` |
| 09 | Pending | ⏳ Queued | Check with `squeue -u mabdel03` |

The most important ones to watch are **Papers 05, 06, and 07** as they're actively training now!

