"""
Utility functions for AGOP experiments analysis

Provides data loading, statistical analysis, and visualization helpers
for analyzing grokking experiments across optimizers and weight decays.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Optional dependency for scipy (for statistical tests)
try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr, ttest_ind
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None
    pearsonr = None
    spearmanr = None
    ttest_ind = None

# Optional dependencies for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None

# Optional dependency for HDF5
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

warnings.filterwarnings('ignore')


def load_experiment(exp_dir: Path) -> Dict:
    """
    Load a single experiment's data
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        Dictionary with keys: 'config', 'history', 'agop' (if available)
    """
    data = {'exp_name': exp_dir.name}
    
    # Load config
    config_path = exp_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            data['config'] = json.load(f)
    else:
        data['config'] = {}
    
    # Load training history
    history_path = exp_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            data['history'] = json.load(f)
    else:
        data['history'] = {}
    
    # Load AGOP metrics from HDF5 if available
    agop_path = exp_dir / 'agop_metrics.h5'
    if agop_path.exists() and HAS_H5PY:
        data['agop'] = {}
        try:
            with h5py.File(agop_path, 'r') as f:
                for key in f.keys():
                    data['agop'][key] = f[key][:]
        except Exception as e:
            print(f"Warning: Could not load AGOP HDF5 from {exp_dir.name}: {e}")
            data['agop'] = {}
    else:
        data['agop'] = {}
    
    return data


def load_all_experiments(dataset_dir: Path, pattern: str = '*') -> Dict[str, Dict]:
    """
    Load all experiments from a dataset directory
    
    Args:
        dataset_dir: Path to dataset results directory
        pattern: Glob pattern for filtering experiments
        
    Returns:
        Dictionary mapping experiment name to experiment data
    """
    experiments = {}
    exp_dirs = sorted(dataset_dir.glob(pattern))
    
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        
        # Only load if has config.json
        if not (exp_dir / 'config.json').exists():
            continue
            
        try:
            exp_data = load_experiment(exp_dir)
            experiments[exp_dir.name] = exp_data
        except Exception as e:
            print(f"Error loading {exp_dir.name}: {e}")
    
    return experiments


def classify_grokking(exp_data: Dict, threshold: float = 0.95, window: int = 10) -> bool:
    """
    Determine if experiment achieved grokking
    
    Args:
        exp_data: Experiment data dictionary
        threshold: Test accuracy threshold for grokking
        window: Number of consecutive epochs to stay above threshold
        
    Returns:
        True if grokked, False otherwise
    """
    history = exp_data.get('history', {})
    test_acc = history.get('test_acc', [])
    
    if not test_acc or len(test_acc) < window:
        return False
    
    # Check if test acc crosses threshold and stays above
    for i in range(len(test_acc) - window + 1):
        if all(acc > threshold for acc in test_acc[i:i+window]):
            return True
    
    return False


def compute_time_to_grok(exp_data: Dict, threshold: float = 0.95, window: int = 10) -> int:
    """
    Calculate epoch when grokking occurred
    
    Args:
        exp_data: Experiment data dictionary
        threshold: Test accuracy threshold for grokking
        window: Number of consecutive epochs to stay above threshold
        
    Returns:
        Grokking epoch, or -1 if no grokking occurred
    """
    history = exp_data.get('history', {})
    test_acc = history.get('test_acc', [])
    epochs = history.get('epoch', list(range(len(test_acc))))
    
    if not test_acc or len(test_acc) < window:
        return -1
    
    # Find first epoch where acc crosses threshold and stays above
    for i in range(len(test_acc) - window + 1):
        if all(acc > threshold for acc in test_acc[i:i+window]):
            return epochs[i]
    
    return -1


def extract_agop_at_epoch(exp_data: Dict, target_epoch: int, 
                          metric_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Extract AGOP metrics at a specific epoch
    
    Args:
        exp_data: Experiment data dictionary
        target_epoch: Target epoch
        metric_names: List of metric names to extract (None = all)
        
    Returns:
        Dictionary mapping metric name to value at target epoch
    """
    history = exp_data.get('history', {})
    epochs = history.get('epoch', [])
    
    if not epochs or target_epoch not in epochs:
        return {}
    
    # Find index of target epoch
    idx = epochs.index(target_epoch)
    
    # Extract metrics from history
    result = {}
    if metric_names is None:
        metric_names = [k for k in history.keys() if k.startswith('agop_')]
    
    for metric in metric_names:
        if metric in history and idx < len(history[metric]):
            result[metric] = history[metric][idx]
    
    return result


def statistical_comparison(group1: List[float], group2: List[float], 
                          labels: Tuple[str, str] = ('Group 1', 'Group 2')) -> Dict:
    """
    Perform statistical comparison between two groups
    
    Args:
        group1: First group of values
        group2: Second group of values
        labels: Labels for the two groups
        
    Returns:
        Dictionary with statistics (means, t-test, effect size)
    """
    if not HAS_SCIPY:
        raise ImportError("Statistical tests require scipy. Install with: pip install scipy")
    
    g1 = np.array(group1)
    g2 = np.array(group2)
    
    # Remove NaN values
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    
    if len(g1) == 0 or len(g2) == 0:
        return {
            'group1_mean': np.nan,
            'group2_mean': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'cohens_d': np.nan,
            'interpretation': 'Insufficient data'
        }
    
    # Compute statistics
    mean1 = np.mean(g1)
    mean2 = np.mean(g2)
    std1 = np.std(g1, ddof=1) if len(g1) > 1 else 0
    std2 = np.std(g2, ddof=1) if len(g2) > 1 else 0
    
    # Welch's t-test (unequal variances)
    if len(g1) > 1 and len(g2) > 1:
        t_stat, p_val = ttest_ind(g1, g2, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan
    
    # Cohen's d effect size
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan
    
    # Interpretation
    if np.isnan(p_val):
        interp = 'N/A'
    elif p_val < 0.001:
        interp = f'Highly significant (p < 0.001), effect size = {cohens_d:.2f}'
    elif p_val < 0.05:
        interp = f'Significant (p = {p_val:.3f}), effect size = {cohens_d:.2f}'
    else:
        interp = f'Not significant (p = {p_val:.3f}), effect size = {cohens_d:.2f}'
    
    return {
        'group1_label': labels[0],
        'group2_label': labels[1],
        'group1_mean': mean1,
        'group1_std': std1,
        'group1_n': len(g1),
        'group2_mean': mean2,
        'group2_std': std2,
        'group2_n': len(g2),
        't_statistic': t_stat,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'interpretation': interp
    }


def generate_summary_table(experiments: Dict[str, Dict], 
                          metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate summary table for experiments
    
    Args:
        experiments: Dictionary of experiment data
        metrics: List of metrics to include (None = standard set)
        
    Returns:
        DataFrame with experiment summaries
    """
    if metrics is None:
        metrics = ['test_acc', 'train_acc', 'grokked', 'grok_epoch']
    
    rows = []
    for exp_name, exp_data in experiments.items():
        config = exp_data.get('config', {})
        history = exp_data.get('history', {})
        
        row = {
            'experiment': exp_name,
            'architecture': config.get('architecture', 'unknown'),
            'optimizer': config.get('optimizer', 'unknown'),
            'weight_decay': config.get('weight_decay', np.nan),
            'lr': config.get('lr', np.nan),
        }
        
        # Add performance metrics
        if 'test_acc' in history and history['test_acc']:
            row['final_test_acc'] = history['test_acc'][-1]
            row['max_test_acc'] = max(history['test_acc'])
        else:
            row['final_test_acc'] = np.nan
            row['max_test_acc'] = np.nan
        
        if 'train_acc' in history and history['train_acc']:
            row['final_train_acc'] = history['train_acc'][-1]
        else:
            row['final_train_acc'] = np.nan
        
        # Grokking info
        row['grokked'] = classify_grokking(exp_data)
        row['grok_epoch'] = compute_time_to_grok(exp_data)
        
        # Add final AGOP metrics if requested
        for metric in metrics:
            if metric.startswith('agop_'):
                if metric in history and history[metric]:
                    row[f'final_{metric}'] = history[metric][-1]
                else:
                    row[f'final_{metric}'] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def smooth_series(x: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing
    
    Args:
        x: Input series
        window: Window size for smoothing
        
    Returns:
        Smoothed series
    """
    smoothed = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        smoothed[i] = np.mean(x[start:i + 1])
    return smoothed


def plot_agop_comparison(experiments: Dict[str, Dict], metric_name: str, 
                        groupby: str = 'optimizer', smooth_window: int = 5,
                        ax: Optional['plt.Axes'] = None, title: Optional[str] = None) -> 'plt.Figure':
    """
    Plot AGOP metric comparison across experiments
    
    Args:
        experiments: Dictionary of experiment data
        metric_name: Name of AGOP metric to plot
        groupby: How to group experiments ('optimizer', 'weight_decay', 'architecture')
        smooth_window: Window for smoothing
        ax: Matplotlib axes (creates new if None)
        title: Plot title (auto-generated if None)
        
    Returns:
        Figure object
    """
    if not HAS_PLOTTING:
        raise ImportError("Plotting requires matplotlib and seaborn. Install with: pip install matplotlib seaborn")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Group experiments
    groups = {}
    for exp_name, exp_data in experiments.items():
        config = exp_data.get('config', {})
        history = exp_data.get('history', {})
        
        if metric_name not in history:
            continue
        
        group_key = config.get(groupby, 'unknown')
        if group_key not in groups:
            groups[group_key] = []
        
        epochs = np.array(history.get('epoch', range(len(history[metric_name]))))
        values = np.array(history[metric_name])
        
        if smooth_window > 1:
            values = smooth_series(values, smooth_window)
        
        groups[group_key].append((epochs, values, exp_name))
    
    # Plot each group
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    for (group_name, group_data), color in zip(groups.items(), colors):
        for i, (epochs, values, exp_name) in enumerate(group_data):
            label = f'{group_name}' if i == 0 else None
            ax.plot(epochs, values, color=color, alpha=0.6, linewidth=1.5, label=label)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    
    if title is None:
        title = f'{metric_name} by {groupby}'
    ax.set_title(title, fontsize=14)
    
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig


def compute_correlation(x: List[float], y: List[float], method: str = 'pearson') -> Tuple[float, float]:
    """
    Compute correlation between two variables
    
    Args:
        x: First variable
        y: Second variable
        method: 'pearson' or 'spearman'
        
    Returns:
        (correlation coefficient, p-value)
    """
    if not HAS_SCIPY:
        raise ImportError("Correlation tests require scipy. Install with: pip install scipy")
    
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    # Remove NaN pairs
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    
    if len(x_clean) < 3:
        return np.nan, np.nan
    
    if method == 'pearson':
        return pearsonr(x_clean, y_clean)
    elif method == 'spearman':
        return spearmanr(x_clean, y_clean)
    else:
        raise ValueError(f"Unknown correlation method: {method}")


def detect_phase_transitions(series: np.ndarray, window: int = 50, 
                             threshold: float = 2.0) -> List[int]:
    """
    Detect phase transitions (sudden changes) in a time series
    
    Args:
        series: Time series data
        window: Window size for detecting changes
        threshold: Z-score threshold for significance
        
    Returns:
        List of indices where transitions detected
    """
    if len(series) < window * 2:
        return []
    
    # Compute local derivatives
    derivatives = np.diff(series)
    
    # Compute rolling statistics
    transitions = []
    for i in range(window, len(derivatives) - window):
        # Compare derivative magnitude to local baseline
        baseline = np.std(derivatives[i-window:i])
        if baseline == 0:
            continue
        
        z_score = abs(derivatives[i]) / baseline
        if z_score > threshold:
            transitions.append(i)
    
    # Merge nearby transitions
    if transitions:
        merged = [transitions[0]]
        for t in transitions[1:]:
            if t - merged[-1] > window // 2:
                merged.append(t)
        return merged
    
    return transitions


def create_comparison_heatmap(experiments: Dict[str, Dict], 
                             metrics: List[str],
                             row_key: str = 'optimizer',
                             col_key: str = 'weight_decay',
                             value_key: str = 'final_test_acc',
                             architecture: Optional[str] = None) -> 'plt.Figure':
    """
    Create heatmap comparing performance across conditions
    
    Args:
        experiments: Dictionary of experiment data
        metrics: List of metrics to include
        row_key: Config key for rows
        col_key: Config key for columns  
        value_key: Metric to display
        architecture: Filter by architecture (None = all)
        
    Returns:
        Figure with heatmap
    """
    if not HAS_PLOTTING:
        raise ImportError("Plotting requires matplotlib and seaborn. Install with: pip install matplotlib seaborn")
    
    # Extract data
    data = []
    for exp_name, exp_data in experiments.items():
        config = exp_data.get('config', {})
        history = exp_data.get('history', {})
        
        if architecture and config.get('architecture') != architecture:
            continue
        
        row_val = config.get(row_key)
        col_val = config.get(col_key)
        
        if value_key in history and history[value_key]:
            value = history[value_key][-1]
        else:
            value = np.nan
        
        data.append({
            'row': row_val,
            'col': col_val,
            'value': value
        })
    
    df = pd.DataFrame(data)
    
    # Pivot to matrix form
    matrix = df.pivot_table(values='value', index='row', columns='col', aggfunc='mean')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': value_key})
    
    arch_str = f' ({architecture})' if architecture else ''
    ax.set_title(f'{value_key} by {row_key} and {col_key}{arch_str}', fontsize=14)
    ax.set_xlabel(col_key.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(row_key.replace('_', ' ').title(), fontsize=12)
    
    plt.tight_layout()
    return fig


def filter_experiments(experiments: Dict[str, Dict], **kwargs) -> Dict[str, Dict]:
    """
    Filter experiments by config parameters
    
    Args:
        experiments: Dictionary of experiment data
        **kwargs: Key-value pairs to filter by (e.g., optimizer='adamw')
        
    Returns:
        Filtered dictionary of experiments
    """
    filtered = {}
    for exp_name, exp_data in experiments.items():
        config = exp_data.get('config', {})
        
        match = True
        for key, value in kwargs.items():
            if config.get(key) != value:
                match = False
                break
        
        if match:
            filtered[exp_name] = exp_data
    
    return filtered

