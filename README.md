# The Geometry of Grok

A comprehensive research project investigating the grokking phenomenon in neural networks through gradient outer product analysis.

## 🎯 Project Goals

Understanding the **geometry of grokking** by analyzing gradient outer products during the transition from memorization to generalization. We replicate 10 major grokking papers and track GOP dynamics across:
- Different datasets (algorithmic, images, text, molecules, knowledge graphs)
- Different architectures (Transformers, MLPs, CNNs, ResNets, Linear networks)
- Different optimizers (Adam, AdamW, SGD)
- Different weight decay settings (0 to 5.0)

## 📁 Repository Structure

```
9.520-The-Geometry-of-Grok/
├── README.md                    # This file
├── Prior_Works.tex              # Literature review of 10 grokking papers
│
├── Replications/                # Code to replicate 10 grokking papers
│   ├── README.md               # Overview of all papers
│   ├── 01_power_et_al_2022_openai_grok/
│   ├── 02_liu_et_al_2022_effective_theory/
│   ├── 03_nanda_et_al_2023_progress_measures/
│   ├── 04_wang_et_al_2024_implicit_reasoners/
│   ├── 05_liu_et_al_2022_omnigrok/
│   ├── 06_humayun_et_al_2024_deep_networks/
│   ├── 07_thilak_et_al_2022_slingshot/
│   ├── 08_doshi_et_al_2024_modular_polynomials/
│   ├── 09_levi_et_al_2023_linear_estimators/
│   └── 10_minegishi_et_al_2023_grokking_tickets/
│
└── New_Explorations/            # GOP analysis framework
    ├── START_HERE.md            # 👈 Quick start guide
    ├── framework/               # Core GOP tracking modules
    ├── configs/                 # Experiment configurations
    ├── experiments/             # Wrapped training scripts
    ├── analysis/                # Visualization and analysis tools
    └── results/                 # Experimental results (created at runtime)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- 50+ GB disk space for testing (TBs for full experiments)
- Access to HPC cluster with SLURM (recommended)

### Setup on HPC Cluster

#### 1. Clone Repository

```bash
cd /om2/user/mabdel03
git clone https://github.com/Mabdel-03/9.520-The-Geometry-of-Grok.git
cd 9.520-The-Geometry-of-Grok
```

#### 2. Create Conda Environment

```bash
# Source conda (required on your cluster)
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh

# Create conda environment
conda create -p /om2/user/mabdel03/conda_envs/SLT_Proj_Env python=3.9 -y

# Activate environment
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Install dependencies for replications
pip install torch torchvision numpy scipy matplotlib h5py pyyaml tqdm psutil

# Install dependencies for GOP analysis
cd New_Explorations
pip install -r requirements.txt
cd ..
```

#### 3. Test the Setup

```bash
# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Run framework validation
cd New_Explorations
python test_framework.py
```

Expected output: `✅ All tests passed! Framework is ready to use.`

### Setup on Local Machine

```bash
git clone https://github.com/Mabdel-03/9.520-The-Geometry-of-Grok.git
cd 9.520-The-Geometry-of-Grok

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy scipy matplotlib h5py pyyaml tqdm psutil
cd New_Explorations
pip install -r requirements.txt
```

## 📚 Two Main Components

### 1. Replications (Step 0)

**Goal:** Replicate 10 grokking papers to observe the phenomenon

**Location:** `Replications/`

**What's Included:**
- 7 cloned GitHub repositories from original papers
- 3 implementations from scratch (for papers without code)
- SLURM batch scripts for each experiment
- Comprehensive documentation

**Read:** `Replications/README.md`

### 2. GOP Analysis Framework (Our Research)

**Goal:** Understand grokking through gradient outer product dynamics

**Location:** `New_Explorations/`

**What It Does:**
- Computes gradient outer product G ⊗ G^T **every epoch**
- Tracks eigenvalues, norms, rank, condition number
- Stores everything in compressed HDF5 format
- Provides visualization and analysis tools

**Read:** `New_Explorations/START_HERE.md`

## 🎯 Recommended Workflow

### Phase 1: Testing (Local or HPC)

```bash
cd New_Explorations

# 1. Validate framework
python test_framework.py

# 2. Run smallest experiment (09_levi_linear: ~1K params, ~6 hours, ~16 GB)
cd experiments/09_levi_linear
sbatch run_gop_analysis.sh  # On HPC
# OR
python wrapped_train.py --config ../../configs/09_levi_linear.yaml  # Locally

# 3. Visualize results
cd ../../analysis
python visualize_gop.py --results_dir ../results/09_levi_linear
```

### Phase 2: Scale Up

Run progressively larger experiments:

```bash
# Medium size (~50K params, ~48 hours, ~50 GB)
cd experiments/08_doshi_polynomials
sbatch run_gop_analysis.sh

# Larger (~100K params, ~72 hours, ~160 GB)
cd ../03_nanda_progress
sbatch run_gop_analysis.sh
```

### Phase 3: Analysis

Compare GOP dynamics across experiments:

```bash
cd analysis

# Compare multiple experiments
python compare_experiments.py \
    --experiments ../results/09_levi_linear ../results/08_doshi_polynomials \
    --save_dir comparison_plots/

# Custom analysis in Python
python
>>> import h5py
>>> with h5py.File('../results/09_levi_linear/metrics.h5', 'r') as f:
...     eigenvalues = f['eigenvalues_top_k'][:]
...     test_acc = f['test_acc'][:]
```

## 📊 What You'll Track

### Every Epoch:
- ✅ Train/test loss and accuracy
- ✅ Full gradient outer product matrix (M × M)
- ✅ Per-layer GOP matrices
- ✅ All eigenvalues + top-k eigenvectors
- ✅ Trace, Frobenius norm, spectral norm, nuclear norm
- ✅ Rank, condition number, determinant
- ✅ Eigenvalue statistics

### Storage Format:
- `metrics.h5` - Scalar time series (~MB)
- `gop_full.h5` - Full GOP matrices (~GB to ~TB)
- `gop_layers.h5` - Per-layer GOPs (~GB)

## 🔧 HPC-Specific Setup

All SLURM scripts are now configured to use your conda environment. They automatically:

1. Source conda from your anaconda installation
2. Activate the environment at `/om2/user/mabdel03/conda_envs/SLT_Proj_Env`
3. Run the experiments

**Note:** You may need to adjust these in each script:
- `#SBATCH --partition=...` - Your cluster's partition name
- `module load cuda/11.8` - Your CUDA version
- Time limits, memory, GPU requirements

## 📖 Papers Replicated

1. **Power et al. (2022)** - Original grokking paper (modular arithmetic)
2. **Liu et al. (2022)** - Effective theory of representation learning
3. **Nanda et al. (2023)** - Mechanistic interpretability (Fourier algorithm)
4. **Wang et al. (2024)** - Knowledge graph reasoning
5. **Liu et al. (2022)** - Omnigrok (beyond algorithmic data)
6. **Humayun et al. (2024)** - Deep networks always grok
7. **Thilak et al. (2022)** - Slingshot mechanism
8. **Doshi et al. (2024)** - Modular polynomials
9. **Levi et al. (2023)** - Linear estimators
10. **Minegishi et al. (2023)** - Lottery tickets accelerate grokking

See `Prior_Works.tex` for detailed literature review.

## 💾 Storage Requirements

### With Full GOP Storage

| Experiment | Parameters | Total Storage (Compressed) |
|-----------|-----------|---------------------------|
| Levi (linear) | 1K | ~16 GB ⭐ |
| Doshi (poly) | 50K | ~50 GB |
| Nanda (transformer) | 100K | ~160 GB |
| Thilak (slingshot) | 150K | ~450 GB |
| Humayun (MNIST) | 160K | ~1 TB |

### Metrics-Only Mode

Set `store_full_gop: false` in config → **< 1 GB per experiment**

Still tracks all eigenvalues and metrics, just not the full matrices.

## 🛠️ Troubleshooting

### Out of Storage

Edit config file:
```yaml
storage:
  store_full_gop: false  # Store only metrics, not full matrices
  
gop_tracking:
  frequency: 10  # Track every 10 epochs instead of every epoch
```

### Out of Memory

Edit config file:
```yaml
gop_tracking:
  use_gpu: false  # Use CPU for GOP computation
```

### Training Too Slow

Reduce tracking frequency:
```yaml
gop_tracking:
  frequency: 100  # Track every 100 epochs
```

## 📈 Expected Results

After running experiments, you'll be able to:

1. **Visualize eigenvalue spectrum evolution** during grokking
2. **Identify GOP signatures** that predict generalization
3. **Compare dynamics** across different setups
4. **Understand the geometry** of the memorization → generalization transition

## 🔬 Research Questions

This framework enables investigating:

- How does the eigenvalue spectrum change during grokking?
- Do different datasets show similar GOP patterns?
- What role do top eigenvalues play in generalization?
- How does weight decay affect GOP dynamics?
- Are there universal GOP signatures of grokking?
- Which layers contribute most to the transition?

## 📚 Documentation

- **Replications:**
  - `Replications/README.md` - Overview of all 10 papers
  - Each paper directory has its own README

- **GOP Framework:**
  - `New_Explorations/START_HERE.md` - Quick start (read first!)
  - `New_Explorations/GETTING_STARTED.md` - Tutorial
  - `New_Explorations/USAGE_GUIDE.md` - Detailed usage
  - `New_Explorations/IMPLEMENTATION_SUMMARY.md` - Technical details

- **Literature:**
  - `Prior_Works.tex` - Comprehensive review of grokking papers

## 🤝 Contributing

### Adding New Experiments

1. Copy template: `cp New_Explorations/framework/template_wrapper.py your_experiment/`
2. Adapt training code
3. Create config from `configs/CONFIG_TEMPLATE.yaml`
4. Test locally, then submit to HPC

### Adding New Analysis

1. Add functions to `New_Explorations/analysis/visualize_gop.py`
2. Or create new analysis scripts
3. HDF5 files are easy to load and analyze

## 📝 Citation

If you use this code, please cite the original papers (see individual README files in `Replications/`) and this repository.

### Original Grokking Papers

See `Prior_Works.tex` for complete citations.

Key papers:
- Power et al. (2022) - Original grokking discovery
- Liu et al. (2022) - Effective theory and Omnigrok
- Nanda et al. (2023) - Mechanistic interpretability

## 📧 Contact

**Course:** MIT 9.520 - The Geometry of Grok  
**Repository:** https://github.com/Mabdel-03/9.520-The-Geometry-of-Grok

## 🎓 Acknowledgments

This project builds upon the excellent work of researchers who investigated grokking:
- OpenAI (Power et al.)
- MIT (Liu, Tegmark, Michaud et al.)
- Anthropic (Nanda et al.)
- OSU NLP Group (Wang et al.)
- And all other authors listed in `Prior_Works.tex`

## 📋 Quick Command Reference

### Setup
```bash
# Clone repo
git clone https://github.com/Mabdel-03/9.520-The-Geometry-of-Grok.git
cd 9.520-The-Geometry-of-Grok

# Setup conda environment (on HPC)
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda create -p /om2/user/mabdel03/conda_envs/SLT_Proj_Env python=3.9 -y
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env
pip install torch torchvision numpy scipy matplotlib h5py pyyaml tqdm psutil
```

### Run Replication (Without GOP Tracking)
```bash
cd Replications/03_nanda_et_al_2023_progress_measures
sbatch run_modular_addition.sh
```

### Run GOP Analysis (With Tracking)
```bash
cd New_Explorations/experiments/09_levi_linear
sbatch run_gop_analysis.sh
```

### Visualize Results
```bash
cd New_Explorations/analysis
python visualize_gop.py --results_dir ../results/09_levi_linear
```

## 🎯 Start Here

**New users:** Read `New_Explorations/START_HERE.md`

**Want to replicate a paper:** Go to `Replications/` and choose a paper

**Want to analyze GOPs:** Go to `New_Explorations/` and follow `GETTING_STARTED.md`

---

**Let's understand the geometry of grokking! 🔬**
