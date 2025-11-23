# Retroactive AGOP Computation Plan

## Strategy: Analyze First, AGOP Later

**Chosen Approach**: Option B - Wait for experiments to finish, analyze grokking behavior, then compute AGOP metrics retroactively from saved checkpoints.

---

## ✅ Why This is Smart

1. **Experiments already running** - No need to restart
2. **Have valuable data now** - 24 Nanda experiments complete with grokking results!
3. **Checkpoints saved** - Can compute AGOP from any saved epoch
4. **More flexible** - Can focus AGOP on interesting epochs (e.g., during grokking transition)
5. **Efficient** - Only compute AGOP for experiments that grokked

---

## 📋 Current Status

### Completed
✅ **24/24 Nanda experiments** - Finished in ~4 minutes each
- 6 experiments show grokking
- All have checkpoints every 1,000 epochs
- Training history saved (loss, accuracy at every 100 epochs)

### Running
🔄 **18/18 MNIST experiments** - Expected completion: 1-2 days
- Will save checkpoints every 5,000 steps
- Training history tracked

---

## 🎯 Phase 1: Analyze Grokking (Now)

**What you can do immediately** with the completed Nanda results:

### 1. Compare Optimizers
```bash
cd /om2/.../optimizer_experiments/analysis
python visualize_spectral_metrics.py \
    --results_dir /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/paper03_nanda \
    --compare \
    --output_dir plots/nanda_comparison
```

### 2. Identify Best Configurations
Already discovered:
- **AdamW + wd 2.0**: Best overall (grokked at epoch 2,600, 100% acc)
- **AdamW + wd 5.0**: Fastest grokking (epoch 900)
- **SGD + wd 0.01**: Only SGD success (epoch 36,800)

### 3. Plot Grokking Curves
```python
import json
import matplotlib.pyplot as plt

experiments = [
    'nanda_adamw_wd1.0',
    'nanda_adamw_wd2.0', 
    'nanda_adamw_wd5.0'
]

plt.figure(figsize=(10, 6))
for exp in experiments:
    with open(f'/om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/paper03_nanda/{exp}/training_history.json') as f:
        data = json.load(f)
    plt.plot(data['epoch'], data['test_acc'], label=exp)

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Grokking Comparison - AdamW with Different Weight Decay')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('grokking_comparison.png')
```

---

## 🔬 Phase 2: Efficient AGOP Computation (Later)

When you're ready to add AGOP metrics, here's the plan:

### Method: Gradient Matrix + Randomized SVD

**Memory Requirements**:
- Nanda: N×M = 3,800 × 226K = **3.4 GB** (✅ Tractable!)
- MNIST: N×M = 1,000 × 200K = **800 MB** (✅ Easy!)

**vs Full AGOP**:
- Nanda: M×M = 226K × 226K = **204 GB** (❌ Intractable)

**60× memory reduction!**

### Implementation

```python
class EfficientAGOPComputer:
    """
    Compute AGOP metrics without storing full matrix.
    
    Key insight: AGOP = (1/N) G^T G where G is gradient matrix (N × M)
    - Top eigenvalues of AGOP = (1/N) × singular values of G²
    - Use torch.svd_lowrank() for top-k only
    """
    
    def compute_from_checkpoint(
        self, 
        checkpoint_path: str,
        data: torch.Tensor,
        labels: torch.Tensor,
        k: int = 20
    ):
        """
        Load checkpoint and compute AGOP metrics.
        
        Memory: O(N × M) + O(k × M) 
        Time: ~60 seconds per checkpoint
        """
        # Load model
        model = load_model_from_checkpoint(checkpoint_path)
        
        # Collect gradient matrix (N × M)
        G = []
        for i in range(len(data)):
            model.zero_grad()
            loss_i = criterion(model(data[i:i+1]), labels[i:i+1])
            loss_i.backward()
            grad_i = torch.cat([p.grad.cpu().view(-1) for p in model.parameters()])
            G.append(grad_i)
        
        G = torch.stack(G)  # (N × M) matrix
        
        # Compute trace (cheap!)
        trace = (G * G).sum().item() / len(data)
        
        # Top-k eigenvalues via randomized SVD
        _, S, _ = torch.svd_lowrank(G, q=k, niter=7)
        eigenvalues = (S ** 2) / len(data)
        
        # All metrics
        return {
            'trace': trace,
            'spectral_radius': eigenvalues[0].item(),
            'eigengap': (eigenvalues[0] - eigenvalues[1]).item(),
            'top_eigenvalue_energy_ratio': eigenvalues[0].item() / trace,
            'spectral_radius_to_trace_ratio': eigenvalues[0].item() / trace,
            'eigenvalues_top_20': eigenvalues.numpy(),
            # Compute top-k energy
            'top_5_energy_ratio': eigenvalues[:5].sum().item() / trace,
            'top_10_energy_ratio': eigenvalues[:10].sum().item() / trace,
            'top_20_energy_ratio': eigenvalues[:20].sum().item() / trace,
            # Effective rank
            'effective_rank': self._compute_effective_rank(eigenvalues, trace),
        }
```

---

## 📅 Timeline for Retroactive AGOP

### Stage 1: Current (Now - Day 2)
- ✅ All Nanda experiments complete
- 🔄 MNIST experiments running
- ✅ Can analyze grokking behavior
- ✅ Identify interesting experiments

### Stage 2: MNIST Complete (Day 1-2)
- Wait for MNIST to finish
- Analyze MNIST grokking
- Identify which experiments to compute AGOP for

### Stage 3: Selective AGOP (Day 3)
**Compute AGOP only for interesting experiments**, e.g.:
- Experiments that grokked (6 Nanda, TBD MNIST)
- Critical epochs: before, during, and after grokking
- Best performing configurations

**Example**:
```python
# For nanda_adamw_wd2.0 (grokked at epoch 2,600)
epochs_to_analyze = [0, 1000, 2000, 2500, 2600, 3000, 5000, 10000, 40000]

for epoch in epochs_to_analyze:
    checkpoint = f'checkpoints/epoch_{epoch}.pt'
    metrics = compute_from_checkpoint(checkpoint, data, labels)
    # Save metrics
```

**Cost**: ~60 seconds × 9 epochs = 9 minutes per experiment  
**Total**: ~1 hour for all 6 grokking experiments

### Stage 4: Full AGOP Analysis (Day 4+)
- Compare AGOP evolution across optimizers
- Test neural collapse hypothesis
- Correlate eigengap with grokking

---

## 🎨 What You Can Do Right Now (Without AGOP)

### 1. Visualize Grokking
All Nanda results are ready!

```bash
cd analysis
python visualize_spectral_metrics.py \
    --results_dir /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/paper03_nanda \
    --compare \
    --output_dir plots
```

### 2. Compare Weight Decay Effect
```python
import json
import matplotlib.pyplot as plt

# Compare AdamW with different weight decays
wds = [0.5, 1.0, 2.0, 5.0]
colors = ['blue', 'green', 'red', 'purple']

plt.figure(figsize=(12, 6))
for wd, color in zip(wds, colors):
    with open(f'/om/scratch/.../paper03_nanda/nanda_adamw_wd{wd}/training_history.json') as f:
        data = json.load(f)
    plt.plot(data['epoch'], data['test_acc'], label=f'WD={wd}', color=color, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Effect of Weight Decay on Grokking (AdamW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('weight_decay_effect.png', dpi=300)
```

### 3. Analyze Grokking Times
```python
grokking_results = {
    'adamw_wd0.5': 12600,
    'adamw_wd1.0': 7500,
    'adamw_wd2.0': 2600,
    'adamw_wd5.0': 900,
    'adamw_wd10.0': 400,
    'sgd_wd0.01': 36800,
}

# Plot weight decay vs grokking speed
wds = [0.5, 1.0, 2.0, 5.0, 10.0]
epochs = [12600, 7500, 2600, 900, 400]

plt.figure(figsize=(10, 6))
plt.plot(wds, epochs, 'o-', markersize=10, linewidth=2)
plt.xlabel('Weight Decay')
plt.ylabel('Grokking Epoch')
plt.yscale('log')
plt.title('Higher Weight Decay → Faster Grokking')
plt.grid(True, alpha=0.3)
plt.savefig('weight_decay_vs_grokking_speed.png', dpi=300)
```

---

## 📊 Current Experiment Status

Run anytime:
```bash
cd /om2/.../optimizer_experiments
./MONITORING_DASHBOARD.sh
```

Or detailed:
```bash
./check_status.py --results_dir /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments
```

---

## 🔮 When Ready for AGOP

**Create an issue/request** and I'll implement:

1. **`framework/agop_efficient.py`** - Gradient matrix method
2. **`analysis/compute_agop_from_checkpoints.py`** - Batch processor
3. **Usage guide** - How to run retroactive computation

**Estimated implementation time**: 2 hours  
**Estimated computation time**: 1-2 hours for all interesting checkpoints  
**Memory required**: 8-16 GB (tractable on any node)

---

## ✨ Summary

**Current Plan**:
1. ✅ Let experiments finish (1-2 days for MNIST)
2. ✅ Analyze grokking behavior (you can start now!)
3. ✅ Identify interesting configurations
4. ⏳ Implement efficient AGOP later (when you're ready)
5. ⏳ Compute AGOP for selected checkpoints
6. ⏳ Correlate AGOP metrics with grokking

**Right now**: Focus on understanding optimizer/weight decay effects on grokking!

**Monitoring**: Run `./MONITORING_DASHBOARD.sh` or `./check_status.py`

---

*You have great grokking data already - enjoy analyzing it while MNIST experiments complete!* 🎉📊

**Next steps when MNIST completes**:
1. Run `./check_status.py` to see all results
2. Visualize everything with the analysis scripts
3. Document findings
4. Then decide if AGOP would add additional insights

