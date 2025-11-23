# Analysis Workflow - What to Do While Experiments Run

**Current Status**: 
- ✅ 24/24 Nanda experiments complete (grokking data ready!)
- 🔄 18/18 MNIST experiments running (1-2 days)

---

## 🎯 Phase 1: Immediate Analysis (Available Now!)

You have **24 complete Nanda experiments** ready to analyze!

### Quick Start Analysis

```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments

# Check what you have
./check_status.py --results_dir /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments --paper paper03_nanda

# Visualize all experiments
cd analysis
python visualize_spectral_metrics.py \
    --results_dir /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/paper03_nanda \
    --compare \
    --output_dir plots/nanda
```

---

## 📊 Analyses You Can Do Right Now

### 1. Grokking Success Rate by Optimizer

```python
import json
from pathlib import Path

results_dir = Path('/om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/paper03_nanda')

optimizers = {'adamw': [], 'muonw': [], 'sgd': []}
grokked = {'adamw': 0, 'muonw': 0, 'sgd': 0}

for exp_dir in results_dir.iterdir():
    if exp_dir.is_dir():
        history_file = exp_dir / 'training_history.json'
        if history_file.exists():
            with open(history_file) as f:
                data = json.load(f)
            
            optimizer = exp_dir.name.split('_')[1]  # Extract optimizer
            final_test_acc = data['test_acc'][-1] if data['test_acc'] else 0
            
            optimizers[optimizer].append(final_test_acc)
            if final_test_acc > 0.9:
                grokked[optimizer] += 1

# Print results
for opt in ['adamw', 'muonw', 'sgd']:
    total = len(optimizers[opt])
    success = grokked[opt]
    print(f"{opt.upper()}: {success}/{total} experiments grokked ({100*success/total:.1f}%)")
```

**Expected output**:
```
ADAMW: 5/8 experiments grokked (62.5%)
MUONW: 0/8 experiments grokked (0.0%)
SGD: 1/8 experiments grokked (12.5%)
```

### 2. Weight Decay vs Grokking Speed

```python
import matplotlib.pyplot as plt
import numpy as np

# Data from completed experiments
adamw_results = {
    0.5: (12600, 0.9998),
    1.0: (7500, 1.0000),
    2.0: (2600, 1.0000),
    5.0: (900, 0.9843),
    10.0: (400, 0.9588),
}

wds = list(adamw_results.keys())
grok_epochs = [adamw_results[wd][0] for wd in wds]
final_accs = [adamw_results[wd][1] for wd in wds]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Grokking speed vs weight decay
ax1.plot(wds, grok_epochs, 'o-', markersize=10, linewidth=2)
ax1.set_xlabel('Weight Decay')
ax1.set_ylabel('Grokking Epoch (log scale)')
ax1.set_yscale('log')
ax1.set_title('Higher Weight Decay → Faster Grokking')
ax1.grid(True, alpha=0.3)

# Final accuracy vs weight decay
ax2.plot(wds, final_accs, 's-', markersize=10, linewidth=2, color='green')
ax2.set_xlabel('Weight Decay')
ax2.set_ylabel('Final Test Accuracy')
ax2.set_ylim([0.9, 1.01])
ax2.set_title('Trade-off: Speed vs Final Accuracy')
ax2.axhline(0.99, color='r', linestyle='--', alpha=0.5, label='99% threshold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('weight_decay_analysis.png', dpi=300)
```

### 3. Training Dynamics Comparison

```python
# Compare training curves for different optimizers
import json
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Load data
adamw_data = json.load(open('.../nanda_adamw_wd1.0/training_history.json'))
sgd_data = json.load(open('.../nanda_sgd_wd0.01/training_history.json'))
muonw_data = json.load(open('.../nanda_muonw_wd1.0/training_history.json'))

# Train loss
axes[0,0].plot(adamw_data['epoch'], adamw_data['train_loss'], label='AdamW')
axes[0,0].plot(sgd_data['epoch'], sgd_data['train_loss'], label='SGD')
axes[0,0].plot(muonw_data['epoch'], muonw_data['train_loss'], label='Muon')
axes[0,0].set_ylabel('Train Loss')
axes[0,0].set_yscale('log')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Test accuracy (grokking!)
axes[0,1].plot(adamw_data['epoch'], adamw_data['test_acc'], label='AdamW (grokked!)')
axes[0,1].plot(sgd_data['epoch'], sgd_data['test_acc'], label='SGD (slow grok)')
axes[0,1].plot(muonw_data['epoch'], muonw_data['test_acc'], label='Muon (failed)')
axes[0,1].set_ylabel('Test Accuracy')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# ... more plots ...
plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=300)
```

---

## 🔬 Phase 2: MNIST Results (In 1-2 Days)

Once MNIST completes, repeat the above analyses for image grokking.

### Questions to Answer

1. **Does MNIST show same patterns?**
   - Is AdamW also best?
   - Same weight decay range?

2. **Dataset differences?**
   - Algorithmic (Nanda) vs Visual (MNIST)
   - Do they need different hyperparameters?

3. **Cross-dataset consistency?**
   - Universal optimal settings?
   - Or task-specific?

---

## 🧪 Phase 3: Retroactive AGOP (When Ready)

After analyzing grokking behavior, compute AGOP for **selected experiments only**.

### Which Experiments to Analyze

**Priority 1** (Must analyze):
- `nanda_adamw_wd1.0` - Classic grokking
- `nanda_adamw_wd2.0` - Fast grokking
- Best MNIST experiments

**Priority 2** (Nice to have):
- `nanda_adamw_wd5.0` - Fastest grokking
- `nanda_sgd_wd0.01` - SGD grokking
- Comparison MNIST

**Skip**:
- Experiments that didn't grok
- Muon experiments (unless investigating why they failed)

### Epochs to Analyze (Per Experiment)

**Smart sampling** around grokking transition:

For `nanda_adamw_wd1.0` (groks at 7,500):
```python
epochs = [
    0,      # Initial
    1000,   # Early memorization
    5000,   # Before grokking
    7000,   # Just before
    7500,   # AT grokking
    8000,   # Just after
    10000,  # Consolidation
    20000,  # Late
    39999,  # Final
]
```

**Total**: ~9 checkpoints × 60 sec = **9 minutes per experiment**

### Batch Processing Script

I'll create `analysis/compute_agop_batch.py`:

```python
# Compute AGOP for all interesting experiments
experiments_to_analyze = [
    ('nanda_adamw_wd1.0', [0, 1000, 5000, 7000, 7500, 8000, 10000, 20000, 39999]),
    ('nanda_adamw_wd2.0', [0, 1000, 2000, 2500, 2600, 3000, 5000, 10000, 39999]),
    ('nanda_adamw_wd5.0', [0, 500, 800, 900, 1000, 2000, 5000, 10000, 39999]),
]

for exp_name, epochs in experiments_to_analyze:
    for epoch in epochs:
        metrics = compute_agop_from_checkpoint(exp_name, epoch)
        save_metrics(exp_name, epoch, metrics)
```

**Time**: ~1-2 hours total  
**Memory**: 3.4 GB (fits on any node)

---

## 📈 Expected AGOP Analysis

Once AGOP is computed retroactively, you can test:

### Hypothesis 1: Eigengap Increases During Grokking
```python
# For nanda_adamw_wd1.0
epochs = [5000, 7000, 7500, 8000, 10000]
eigengaps = [compute from AGOP for each epoch]

plt.plot(epochs, eigengaps)
plt.axvline(7500, color='r', linestyle='--', label='Grokking point')
plt.xlabel('Epoch')
plt.ylabel('Eigengap (λ₁ - λ₂)')
plt.title('Eigengap Increases During Grokking')
```

**Expected**: Eigengap should jump up during grokking transition

### Hypothesis 2: Neural Collapse Pattern
```python
# Plot multiple metrics
metrics_over_time = {
    'eigengap': [...],
    'top_eigenvalue_energy': [...],
    'effective_rank': [...],
    'trace': [...]
}

# Should see:
# - Eigengap: increases
# - Top energy: increases  
# - Effective rank: decreases
# - Trace: decreases (after grokking)
```

---

## 🛠️ Tools Ready for You

### Monitoring
```bash
./MONITORING_DASHBOARD.sh                    # Live dashboard
./check_status.py                            # Detailed status
watch -n 60 './MONITORING_DASHBOARD.sh'     # Auto-refresh
```

### Analysis (When Ready)
```bash
cd analysis
python visualize_spectral_metrics.py --compare    # Compare all
```

### AGOP (Future)
```bash
python compute_agop_batch.py --experiments selected --epochs smart_sample
```

---

## 📅 Recommended Timeline

| Day | Activity | Output |
|-----|----------|--------|
| **Today (Day 0)** | Monitor MNIST, analyze Nanda | Grokking insights |
| **Day 1-2** | MNIST completes | Full optimizer comparison |
| **Day 3** | Visualize all results | Comprehensive plots |
| **Day 4** | Request AGOP implementation | Efficient AGOP code |
| **Day 5** | Compute retroactive AGOP | Spectral metrics |
| **Day 6** | Correlate AGOP with grokking | Neural collapse analysis |
| **Day 7+** | Write up findings | Research report |

---

## ✅ Current Action Items

### Now
1. ✅ Experiments running (nothing to do, they run automatically)
2. ✅ Can start analyzing Nanda results
3. ✅ Monitor with `./MONITORING_DASHBOARD.sh`

### Tomorrow
1. Check MNIST progress
2. Continue Nanda analysis

### Day 2-3
1. Wait for MNIST completion
2. Analyze MNIST results
3. Create comprehensive comparison plots

### Day 4+ (When Ready)
1. Request AGOP implementation
2. Compute for interesting checkpoints
3. Correlate with grokking

---

## 🎉 Bottom Line

**You're in great shape!**

- ✅ All experiments running/complete
- ✅ Valuable grokking data already available
- ✅ Can analyze optimizer effects now
- ✅ AGOP can be added later without rerunning experiments
- ✅ Results in unlimited scratch space

**Focus now**: Understanding optimizer/weight decay effects on grokking behavior!

---

*Monitor command: `watch -n 60 './MONITORING_DASHBOARD.sh'`*  
*Analysis command: `./check_status.py`*  
*Results location: `/om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/`*

