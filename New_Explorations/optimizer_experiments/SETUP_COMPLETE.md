# Optimizer Comparison Framework - Setup Complete! ✓

## What Has Been Created

A comprehensive framework for studying grokking with different optimizers, weight decay values, and spectral metrics tracking.

### 📁 Directory Structure

```
optimizer_experiments/
├── framework/                     ✓ Core implementation
│   ├── spectral_metrics.py       ✓ Computes all requested metrics
│   ├── muon_optimizer.py         ✓ Muon & MuonW optimizer
│   ├── trainer.py                ✓ Unified training framework
│   └── __init__.py               ✓
│
├── paper03_nanda/                 ✓ Nanda et al. experiments
│   └── train_nanda.py            ✓ Full implementation
│
├── paper05_omnigrok/              ✓ Omnigrok MNIST experiments
│   └── train_mnist.py            ✓ Full implementation
│
├── configs/                       ✓ Configuration files
│   └── weight_decay_sweep.yaml   ✓
│
├── slurm_scripts/                 ✓ SLURM job scripts
│   ├── run_nanda_single.sh       ✓ Single Nanda job
│   ├── run_mnist_single.sh       ✓ Single MNIST job
│   ├── run_all_nanda.sh          ✓ All Nanda combinations
│   └── run_all_mnist.sh          ✓ All MNIST combinations
│
├── analysis/                      ✓ Analysis tools
│   └── visualize_spectral_metrics.py  ✓ Comprehensive visualization
│
├── README.md                      ✓ Full documentation
├── QUICK_START.md                 ✓ Getting started guide
├── requirements.txt               ✓ Dependencies
└── check_status.py                ✓ Monitor experiments
```

## 🎯 Features Implemented

### 1. Spectral Metrics (ALL REQUESTED METRICS)
✓ **Eigengap**: λ₁ - λ₂  
✓ **Top-k subspace**: Energy in top-k eigenvectors (k=5,10,20,50)  
✓ **Energy concentration in top eigenvector**: λ₁/Σλᵢ  
✓ **Spectral radius**: λ_max  
✓ **Trace**: Σλᵢ  
✓ **Spectral radius to trace ratio**: λ_max/Σλᵢ  

### 2. Optimizers
✓ **Muon**: Momentum-based orthogonal updates  
✓ **MuonW**: Muon with decoupled weight decay  
✓ **Adam/AdamW**: Standard adaptive learning  
✓ **SGD**: Stochastic gradient descent with momentum  

### 3. Experiments
✓ **Paper 3 (Nanda)**: Modular addition, 1-layer ReLU Transformer  
✓ **Paper 5 (Omnigrok)**: MNIST grokking, 3-layer MLP  
⚠ **Paper 4 (Wang)**: Compositional reasoning (complex, deferred)  

### 4. Weight Decay Sweep
✓ Testing: 0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0  
✓ Automated submission scripts  
✓ Total: 24 Nanda experiments + 18 MNIST experiments  

### 5. Analysis Tools
✓ Training curve visualization  
✓ Spectral metrics plotting (6-panel)  
✓ Top-k eigenvalue evolution  
✓ Multi-experiment comparison  
✓ Status checking utility  

## 🚀 How to Use

### Quick Test
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments

# Test Nanda experiment
cd paper03_nanda
python train_nanda.py --optimizer adamw --weight_decay 1.0 --n_epochs 5000 --experiment_name test

# Test MNIST experiment
cd ../paper05_omnigrok
python train_mnist.py --optimizer adamw --weight_decay 0.01 --n_epochs 10000 --experiment_name test
```

### Full Run (SLURM)
```bash
cd slurm_scripts

# Submit all experiments
bash run_all_nanda.sh    # 24 jobs
bash run_all_mnist.sh    # 18 jobs

# Monitor
squeue -u $USER
./check_status.py --results_dir results
```

### Analyze Results
```bash
# After experiments complete
cd analysis
python visualize_spectral_metrics.py --results_dir ../results/paper03_nanda --compare
```

## 📊 Expected Outputs

For each experiment (e.g., `nanda_adamw_wd1.0`):
- `config.json` - Experiment configuration
- `training_history.json` - Train/test loss and accuracy
- `spectral_metrics.h5` - ALL spectral metrics (HDF5 format)
- `checkpoints/` - Model checkpoints

Plots generated:
- `training_curves.png` - Train/test loss and accuracy
- `spectral_metrics.png` - 6-panel spectral analysis
- `top_eigenvalues.png` - Top-10 eigenvalue evolution
- `comparison_*.png` - Multi-experiment comparisons

## 🔬 Research Questions Enabled

1. **Optimizer Comparison**: Which optimizer groks fastest?
2. **Weight Decay Effect**: Optimal weight decay for each optimizer?
3. **Spectral Signatures**: Do spectral metrics predict grokking?
4. **Universal Patterns**: Similar patterns across datasets?
5. **Early Indicators**: Can we predict grokking early?

## ⚙️ Technical Details

### Storage Requirements
- **Per Nanda experiment**: ~500 MB (40K epochs, spectral freq=100)
- **Per MNIST experiment**: ~1 GB (100K steps, spectral freq=100)
- **Total for all**: ~50-100 GB

### Compute Time (Estimated)
- **Nanda**: 6-12 hours per experiment (1 GPU)
- **MNIST**: 12-24 hours per experiment (1 GPU)
- **Parallel**: All can run simultaneously on cluster

### Memory Requirements
- **Nanda**: 8-16 GB GPU memory
- **MNIST**: 16-32 GB GPU memory

## 📚 Documentation

- **README.md**: Comprehensive documentation with examples
- **QUICK_START.md**: Step-by-step getting started guide
- **Code comments**: Detailed docstrings in all Python files
- **Config files**: YAML configuration with comments

## ✅ What Works

- ✓ Spectral metrics computation (tested with toy examples)
- ✓ Muon optimizer implementation
- ✓ Training framework with all optimizers
- ✓ Data loading for Nanda and MNIST
- ✓ Model architectures from original papers
- ✓ HDF5 storage for efficient metric saving
- ✓ Visualization scripts
- ✓ SLURM submission system
- ✓ Status monitoring

## ⚠️ Notes and Limitations

### Paper 4 (Wang et al.)
- **Status**: Framework created but full integration pending
- **Reason**: Complex dependencies (simpletransformers, custom GPT-2)
- **Workaround**: Can be added later if needed
- **Priority**: Papers 3 and 5 cover both algorithmic and image domains

### Computational Costs
- Computing full GOP matrices is expensive for large models
- **Solution**: Adjustable `spectral_freq` parameter
- **Fallback**: Compute only top-k eigenvalues
- **Recommendation**: Start with spectral_freq=100, increase if needed

### Spectral Metrics Frequency
- Default: Every 100 epochs
- Can be adjusted: `--spectral_freq 500` for faster training
- Trade-off: Temporal resolution vs. compute time

## 🎓 Learning from Existing Implementations

The framework leverages working code from:
- `/Replications/03_nanda_et_al_2023_progress_measures/` - Model architecture
- `/Replications/05_liu_et_al_2022_omnigrok/` - MNIST setup
- `/New_Explorations/framework/` - Existing GOP infrastructure

## 🔧 Customization

### Add a New Optimizer
```python
# In framework/trainer.py, add to _create_optimizer():
elif self.optimizer_name == 'my_optimizer':
    return MyOptimizer(params, lr=self.lr, ...)
```

### Add a New Metric
```python
# In framework/spectral_metrics.py, add to compute_metrics():
metrics['my_metric'] = compute_my_metric(eigenvalues)
```

### Modify Weight Decay Values
```yaml
# In configs/weight_decay_sweep.yaml:
weight_decay_values:
  - 0.05
  - 0.15
  # ... your values
```

## 📧 Support

- Check README.md for detailed documentation
- Review QUICK_START.md for step-by-step instructions
- Inspect code comments in framework/ for implementation details
- Refer to `/Replications/` for original paper implementations

## 🎉 Ready to Run!

Everything is set up and ready to go. To start:

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Submit jobs**: `cd slurm_scripts && bash run_all_nanda.sh`
3. **Monitor progress**: `./check_status.py`
4. **Analyze results**: `cd analysis && python visualize_spectral_metrics.py --compare`

**Happy experimenting!** 🧪✨

---

*Framework created: November 2025*  
*Location: `/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/`*

