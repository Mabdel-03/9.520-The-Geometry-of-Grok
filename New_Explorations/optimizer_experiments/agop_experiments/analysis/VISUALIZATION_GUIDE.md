# Enhanced AGOP Visualization Guide

## Overview

The enhanced visualization system provides **9 comprehensive plots** for analyzing AGOP metrics and their relationship to grokking. Key features include:

- 🎯 **Automatic grokking detection** - Identifies when test accuracy crosses 95% threshold
- 📊 **Dual-axis alignment** - Shows exact correlation between test accuracy and AGOP metrics  
- 📈 **Comprehensive timelines** - All metrics on shared x-axis with grokking markers
- 🔍 **Smoothing** - Reduces noise for clearer trends

## Quick Start

```bash
# Single experiment - creates 9 plots
python visualize_agop_metrics.py \
    --results_dir ../results/agop_experiments/nanda/nanda_adamw_wd1.0_seed42

# Compare multiple experiments
python visualize_agop_metrics.py \
    --results_dir ../results/agop_experiments/nanda \
    --experiment_pattern "nanda_*_wd1.0_*" \
    --compare_optimizers

# Test with synthetic data
python test_visualizations.py
```

## Generated Visualizations

### 1. Basic Plots (Always Generated)

#### `training_curves.png`
- **Purpose**: Standard accuracy monitoring
- **Content**: Train and test accuracy over epochs
- **Use case**: Quick check if grokking occurred

#### `agop_basic_metrics.png` (3 panels)
- **Panel 1**: Frobenius norm ||AGOP||_F
- **Panel 2**: Spectral radius (λ₁) - largest eigenvalue
- **Panel 3**: Trace (Σλᵢ) - total variance
- **Use case**: Monitor overall AGOP magnitude and scale

#### `agop_collapse_metrics.png` (2 panels)
- **Panel 1**: Eigengap (λ₁ - λ₂) - gradient alignment
- **Panel 2**: Variation Collapse Ratio (λ₁/Σλᵢ) - concentration measure
- **Use case**: Track collapse toward dominant eigenvector

#### `agop_subspace_similarity.png`
- **Content**: Top-k subspace stability over time
- **Use case**: Detect sudden changes in gradient directions

### 2. Enhanced Visualizations (Single Experiment Only)

#### `combined_grokking_agop.png` (2×2 grid)
- **Top-left**: Test accuracy with grokking threshold
- **Top-right**: Variation collapse ratio
- **Bottom-left**: Eigengap
- **Bottom-right**: Spectral radius vs trace
- **Use case**: Overview of grokking and AGOP evolution

#### `comprehensive_timeline.png` (5-panel shared x-axis) ⭐ NEW
- **Panel 1**: Train/test accuracy with **grokking annotation**
- **Panel 2**: Variation collapse ratio
- **Panel 3**: Eigengap
- **Panel 4**: Spectral radius vs trace
- **Panel 5**: Subspace similarity
- **Features**:
  - Red dashed line marks grokking epoch across all panels
  - Shared x-axis for precise alignment
  - Annotation box shows grokking epoch
- **Use case**: Publication-quality figure showing complete story

#### `aligned_test_acc_vcr.png` ⭐ NEW
- **Left axis (blue)**: Test accuracy
- **Right axis (red)**: Variation collapse ratio (smoothed)
- **Features**:
  - Grokking threshold line
  - Grokking epoch annotation
  - Red dot on metric at grokking point
- **Use case**: See exact correlation between generalization and VCR

#### `aligned_test_acc_eigengap.png` ⭐ NEW
- **Left axis**: Test accuracy
- **Right axis**: Eigengap (λ₁ - λ₂)
- **Use case**: Track gradient alignment during grokking

#### `aligned_test_acc_trace.png` ⭐ NEW
- **Left axis**: Test accuracy  
- **Right axis**: Trace (total variance)
- **Use case**: Monitor gradient magnitude changes

#### `aligned_test_acc_spectral.png` ⭐ NEW
- **Left axis**: Test accuracy
- **Right axis**: Spectral radius (λ₁)
- **Use case**: Track dominant gradient direction strength

## Key Features Explained

### Automatic Grokking Detection

The system automatically detects when grokking occurs:

```python
def detect_and_annotate_grokking(ax, epochs, test_acc, threshold=0.95, window=10):
    """
    Finds first epoch where test accuracy crosses threshold and stays above
    for at least `window` epochs (for stability).
    """
```

**Detection criteria**:
- Test accuracy > 95% (configurable)
- Stays above threshold for 10 consecutive epochs (avoids false positives)

**Visual markers**:
- Red vertical dashed line at grokking epoch
- Text annotation showing epoch number
- Red dot on AGOP metric at grokking point (dual-axis plots)

### Smoothing

All AGOP metrics use moving average smoothing:

```python
def smooth_series(x, window=5):
    """
    Apply moving average over last `window` values.
    Reduces noise without losing trend information.
    """
```

**Default window**: 5 epochs
**Why smooth?**: AGOP metrics can be noisy due to:
- Finite sample approximation
- Stochastic gradient updates
- Numerical precision in eigendecomposition

### Dual-Axis Plots

Perfect for answering: "Does metric X change **when** grokking happens?"

**Advantages**:
- Two different scales on same plot
- Precise temporal alignment
- Easy to spot correlations
- Publication-ready

**How to interpret**:
1. Look for changes in red line (AGOP metric) near blue threshold crossing (grokking)
2. Red dot marks metric value at exact grokking epoch
3. Simultaneous changes suggest causal relationship

## Research Questions You Can Answer

### 1. Does VCR predict grokking?
**Plot**: `aligned_test_acc_vcr.png`
**Look for**: VCR increasing **before** test accuracy rises

### 2. Does eigengap collapse or expand during grokking?
**Plot**: `aligned_test_acc_eigengap.png`  
**Look for**: Sharp change at grokking epoch

### 3. Do gradients concentrate into single direction?
**Plot**: `comprehensive_timeline.png`, panels 2-4
**Look for**: VCR increasing while trace decreases

### 4. Is there a phase transition?
**Plot**: All aligned plots
**Look for**: Sudden simultaneous changes across metrics

### 5. Different patterns for different optimizers?
**Workflow**:
```bash
# Generate plots for each optimizer
for opt in adamw muon sgd; do
    python visualize_agop_metrics.py \
        --results_dir results/nanda/nanda_${opt}_wd1.0_seed42
done

# Then compare visually
```

## Examples

### Example 1: Successful Grokking

Expected pattern in `comprehensive_timeline.png`:
1. **Panel 1**: Test acc jumps from ~10% to 99% around epoch 5000
2. **Panel 2**: VCR increases from 0.2 to 0.8 around same time
3. **Panel 3**: Eigengap increases (gradient alignment)
4. **Panel 4**: Trace decreases (lower variance), spectral radius stabilizes
5. **Panel 5**: Subspace similarity stays high (stable directions)

### Example 2: No Grokking

Expected pattern:
1. **Panel 1**: Test acc stays low (<50%)
2. **Panel 2**: VCR remains flat around 0.3
3. **Panel 3**: Eigengap doesn't increase significantly
4. **Panel 4**: Trace stays high, spectral radius fluctuates
5. **Panel 5**: Lower subspace similarity (unstable)

## Testing

Test the visualizations with synthetic data:

```bash
cd analysis/
python test_visualizations.py
```

This creates:
- Synthetic grokking experiment (test acc rises at epoch 5000)
- Synthetic non-grokking experiment (test acc stays low)
- Generates all 9 plots for both

**Output location**: Check terminal output for temp directory path

## Troubleshooting

### Missing AGOP data
**Error**: "No AGOP data available"
**Solution**: Ensure AGOP was computed during training (`--agop_freq` argument)

### No grokking detected
**Behavior**: No red vertical line on plots
**Reasons**:
- Test accuracy never crossed 95%
- Not enough epochs (grokking didn't occur yet)
- Window size too strict (try reducing in code)

### Plots look noisy
**Solution**: Increase smoothing window:
```python
# In visualize_agop_metrics.py, change:
window = 5  # to larger value, e.g., 10
```

### Dual-axis scales look wrong
**Cause**: Extreme values in AGOP metrics
**Solution**: Metrics are auto-scaled. If needed, manually adjust y-limits in code

## Advanced Usage

### Custom Metrics

Add new AGOP metrics to dual-axis plots:

```python
# In main() function, add:
plot_aligned_dual_axis(
    exp_data,
    'agop_custom_metric',  # Your metric name from HDF5
    'Custom Metric Label',
    output_dir / 'aligned_test_acc_custom.png'
)
```

### Change Grokking Threshold

```python
# In detect_and_annotate_grokking calls:
grok_epoch = detect_and_annotate_grokking(
    ax, epochs, test_acc, 
    threshold=0.90,  # Lower threshold
    window=20        # More epochs for stability
)
```

### Export for Publications

All plots save at 300 DPI with tight bounding boxes:
```python
plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

**File formats supported**: PNG (default), PDF, SVG
```bash
# Change extension in script for vector graphics:
save_path = output_dir / 'plot.pdf'  # or .svg
```

## References

- **Implementation**: Based on `Group1_Grokking_Code_Base.ipynb` Cells 7-9
- **AGOP Theory**: Beaglehole et al. "Average gradient outer product as a mechanism for deep neural collapse"
- **Visualization Design**: Tufte principles of data visualization

## Support

For issues or questions:
1. Check `README.md` in parent directory
2. Review `AGOP_QUICK_REFERENCE.md` for metric definitions
3. Test with synthetic data first (`test_visualizations.py`)

