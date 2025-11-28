# AGOP Experiments Analysis Infrastructure

Comprehensive analysis tools for comparing optimizer and weight decay effects on grokking across all completed experiments (60+ with full AGOP metrics).

## Overview

This analysis infrastructure provides:
1. **Data loading and preprocessing utilities**
2. **Statistical comparison functions**
3. **Interactive Jupyter notebooks** for detailed analysis
4. **Automated figure generation**

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies for full functionality
pip install matplotlib seaborn scipy h5py nbformat

# Minimal install (data loading only)
pip install numpy pandas
```

### 2. Run Tests

```bash
cd analysis/
python3 test_basic_functionality.py
```

This validates:
- ✓ Experiments load correctly
- ✓ Summary tables generate
- ✓ Grokking classification works
- ✓ Statistical functions available (if scipy installed)
- ✓ Notebooks exist

### 3. Run Analysis Notebooks

```bash
jupyter notebook
```

Then open:
- `analyze_nanda_experiments.ipynb` - Nanda modular addition analysis
- `analyze_softmax_experiments.ipynb` - Softmax modular addition analysis
- `cross_dataset_comparison.ipynb` - Cross-dataset comparison

## Files Created

### Core Utilities

- **`analysis_utils.py`** - Main utility module with functions for:
  - `load_all_experiments()` - Batch load experimental results
  - `generate_summary_table()` - Create pandas DataFrame summaries
  - `classify_grokking()` - Detect grokking (test acc > 95%)
  - `compute_time_to_grok()` - Find grokking epoch
  - `statistical_comparison()` - T-tests with effect sizes
  - `filter_experiments()` - Filter by config parameters
  - `smooth_series()` - Moving average smoothing
  - `compute_correlation()` - Pearson/Spearman correlations
  - `plot_agop_comparison()` - Plot AGOP metrics across conditions
  - `create_comparison_heatmap()` - Heatmap visualizations

### Analysis Notebooks

- **`analyze_nanda_experiments.ipynb`** (14.3 KB)
  - Section A: Optimizer Comparison (AdamW, Muon, SGD)
  - Section B: Weight Decay Effects (0.1, 1.0, 5.0, 10.0)
  - Section C: AGOP Metrics Analysis
  - Grokking vs non-grokking comparisons
  - Statistical tests
  - 20 experiments analyzed

- **`analyze_softmax_experiments.ipynb`** (4.8 KB)
  - Same structure as Nanda
  - Special focus on Muon breakthrough
  - 19 experiments analyzed
  - Higher grokking rate (37.5% vs 8.3%)

- **`cross_dataset_comparison.ipynb`** (5.1 KB)
  - Compare Nanda vs Softmax
  - Why Softmax groks better
  - Muon analysis across datasets
  - Dataset-specific AGOP signatures

### Testing Scripts

- **`test_basic_functionality.py`** - Validates core infrastructure
- **`generate_analysis_notebooks.py`** - Notebook generation script

### Directory Structure

```
analysis/
├── analysis_utils.py              # Core utilities
├── analyze_nanda_experiments.ipynb
├── analyze_softmax_experiments.ipynb  
├── cross_dataset_comparison.ipynb
├── generate_analysis_notebooks.py
├── test_basic_functionality.py
├── README_ANALYSIS.md             # This file
└── figures/                       # Generated figures
    ├── nanda/
    ├── softmax/
    └── cross_dataset/
```

## Current Experiment Status

### All Datasets
- **Total experiments:** 72 directories created
- **Complete with AGOP data:** 60/72 ✅
- **Analysis notebooks:** 3 comprehensive Jupyter notebooks

### Dataset-Specific Status
- **Nanda:** 24 experiments (modular addition, ReLU transformer)
- **Softmax:** 24 experiments (modular addition, standard transformer)
- **MNIST:** 12 experiments (image classification, MLP)
- **Composition:** 12 experiments (compositional reasoning)

### Key Findings from Completed Experiments
1. **🎉 Major Discovery:** Muon groks on Softmax transformers with one-hot inputs!
2. **Transformers >> MLPs:** ~2.4× better grokking rate across datasets
3. **AdamW most reliable:** 75% success rate on transformer architectures
4. **Softmax shows highest grokking rate:** ~47% vs ~20% for Nanda
5. **One-hot encoding matters:** Enables new optimizer behaviors (Muon success)

## Analysis Workflow

### For Nanda Dataset

```python
from analysis_utils import *

# Load experiments
experiments = load_all_experiments(Path('../results/nanda'))

# Generate summary
summary_df = generate_summary_table(experiments)
print(summary_df)

# Filter by optimizer
adamw_exps = filter_experiments(experiments, optimizer='adamw')

# Classify grokking
for name, exp in adamw_exps.items():
    if classify_grokking(exp):
        grok_epoch = compute_time_to_grok(exp)
        print(f"{name}: grokked at epoch {grok_epoch}")

# Statistical comparison
adamw_acc = [exp['history']['test_acc'][-1] for exp in adamw_exps.values()]
muon_exps = filter_experiments(experiments, optimizer='muon')
muon_acc = [exp['history']['test_acc'][-1] for exp in muon_exps.values()]

stats = statistical_comparison(adamw_acc, muon_acc, ('AdamW', 'Muon'))
print(stats['interpretation'])
```

### In Jupyter Notebooks

The notebooks provide:
- **Interactive exploration** with pandas DataFrames
- **Automated visualizations** (test accuracy curves, AGOP evolution)
- **Statistical tests** with interpretation
- **Publication-quality figures** saved to `figures/` directory

## AGOP Metrics Analyzed

1. **Variation Collapse Ratio (VCR):** λ₁ / Σλᵢ
2. **Eigengap:** λ₁ - λ₂  
3. **Trace:** Σλᵢ
4. **Spectral Radius:** λ₁
5. **Top Eigenvalue Energy**
6. **Subspace Similarity**

## Comparison Structure

### Section A: Optimizer Comparison
- Within MLP architecture: AdamW vs Muon vs SGD
- Within Transformer architecture: AdamW vs Muon vs SGD
- Statistical tests (Welch's t-test)
- Effect sizes (Cohen's d)

### Section B: Weight Decay Comparison  
- Within each optimizer: effect of different WD values
- Correlation analysis (Spearman)
- Optimal WD identification

### Section C: AGOP Metrics Analysis
1. **Grokking vs Non-Grokking:**
   - Mean ± std evolution plots
   - Divergence point identification
   - Statistical tests at key epochs

2. **Optimizer-Specific AGOP Patterns:**
   - VCR evolution by optimizer
   - Eigengap dynamics
   - Trace changes

3. **Weight Decay Effects on AGOP:**
   - How WD affects AGOP magnitudes
   - Optimal WD correlation with AGOP values

4. **Phase Transition Detection:**
   - Sudden AGOP changes
   - Alignment with grokking epochs

## Dependencies

### Required
- `numpy` - Array operations
- `pandas` - Data manipulation

### Optional (for full functionality)
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `scipy` - Statistical tests
- `h5py` - AGOP metrics from HDF5
- `nbformat` - Notebook generation

**Note:** The infrastructure gracefully handles missing optional dependencies.

## Troubleshooting

### Missing Dependencies
If you see import errors, install missing packages:
```bash
pip install <package-name>
```

### No AGOP Data
Some experiments may not have `agop_metrics.h5` files. The analysis will skip AGOP-specific analyses for these experiments but still process accuracy data.

### Notebooks Won't Run
Ensure you're in the correct directory and have Jupyter installed:
```bash
pip install jupyter
cd analysis/
jupyter notebook
```

## Future Enhancements

Potential additions:
- [ ] Automated report generation (PDF/HTML)
- [ ] Interactive dashboards (Plotly/Dash)
- [ ] Time-series change-point detection
- [ ] Logistic regression (grokking prediction from AGOP)
- [ ] Cross-validation of findings on additional datasets

## References

- **AGOP Theory:** Beaglehole et al. "Average gradient outer product as a mechanism for deep neural collapse"
- **Grokking:** Power et al. "Grokking: Generalization beyond overfitting on small algorithmic datasets"
- **Experimental Setup:** Based on Group1_Grokking_Code_Base.ipynb

## Support

For questions or issues:
1. Check `README.md` in parent directory
2. Review `VISUALIZATION_GUIDE.md` for plotting details
3. Run `test_basic_functionality.py` to validate setup
4. Check experiment logs in `../slurm_scripts/logs/`

---

**Created:** November 26, 2024  
**Last Updated:** November 28, 2024  
**Analysis Infrastructure Version:** 1.1  
**Total Experiments Available:** 60+ with complete AGOP metrics  
**Analysis Tools:** 3 Jupyter notebooks, Python utilities, automated visualization pipeline

