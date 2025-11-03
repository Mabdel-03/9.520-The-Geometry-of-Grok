# Complete Setup Instructions for HPC Cluster

This guide provides step-by-step instructions for setting up the grokking research environment on your HPC cluster.

## Prerequisites

- Access to `/om2/user/mabdel03/` on your HPC cluster
- Anaconda installed at `/om2/user/mabdel03/anaconda/`
- SLURM job scheduler
- CUDA-capable GPUs

## Step-by-Step Setup

### 1. Clone the Repository

```bash
# SSH into your HPC cluster
ssh your-cluster-address

# Navigate to your home directory
cd /om2/user/mabdel03

# Clone the repository
git clone https://github.com/Mabdel-03/9.520-The-Geometry-of-Grok.git

# Navigate to the project
cd 9.520-The-Geometry-of-Grok
```

### 2. Create Conda Environment

**Option A: Automated Setup (Recommended)**

```bash
# Make setup script executable
chmod +x setup_conda_env.sh

# Run the setup script
./setup_conda_env.sh
```

The script will:
- Create conda environment at `/om2/user/mabdel03/conda_envs/SLT_Proj_Env`
- Install all required packages (PyTorch, NumPy, SciPy, h5py, etc.)
- Verify installation

**Option B: Manual Setup**

```bash
# Source conda
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh

# Create environment
conda create -p /om2/user/mabdel03/conda_envs/SLT_Proj_Env python=3.9 -y

# Activate environment
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install scientific computing packages
pip install numpy scipy matplotlib pandas scikit-learn

# Install data handling
pip install h5py pyyaml tqdm psutil

# Install additional packages
pip install jupyter ipython tensorboard transformers datasets networkx

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 3. Test the Setup

#### Test Framework Validation

```bash
# Activate conda
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to GOP framework
cd New_Explorations

# Run validation tests
python test_framework.py
```

**Expected output:**
```
✓ GOP computation test PASSED
✓ Trace test PASSED
✓ Eigenvalue sum test PASSED
✓ Frobenius norm test PASSED
✓ Storage test PASSED
✓ Memory estimation test PASSED

✅ END-TO-END TEST PASSED!
🎉 All tests passed! Framework is ready to use.
```

### 4. Submit Your First Job

Start with the smallest experiment (Linear Estimators):

```bash
cd experiments/09_levi_linear

# Review the SLURM script
cat run_gop_analysis.sh

# Submit job
sbatch run_gop_analysis.sh

# Check job status
squeue -u mabdel03

# Monitor output (replace JOBID with actual job ID)
tail -f logs/gop_linear_JOBID.out
```

### 5. Check Results

```bash
# After job completes, check results directory
ls -lh ../../results/09_levi_linear/

# You should see:
# - metrics.h5 (scalar time series)
# - gop_full.h5 (GOP matrices)
# - gop_layers.h5 (per-layer GOPs)
# - config.json (experiment configuration)
```

### 6. Visualize Results

```bash
cd ../../analysis

python visualize_gop.py --results_dir ../results/09_levi_linear

# Check the plots directory
ls ../results/09_levi_linear/plots/
```

## Environment Activation for Future Sessions

Every time you log in to the cluster, activate the environment:

```bash
# Add to your ~/.bashrc for automatic activation
echo 'source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh' >> ~/.bashrc
echo 'alias activate_grok="conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env"' >> ~/.bashrc

# Or activate manually each session
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env
```

## SLURM Script Configuration

All SLURM scripts in this repository are pre-configured to:
1. Source conda from `/om2/user/mabdel03/anaconda/etc/profile.d/conda.sh`
2. Activate environment at `/om2/user/mabdel03/conda_envs/SLT_Proj_Env`
3. Load CUDA module

**You may need to adjust:**
- CUDA version: `module load cuda/11.8` → match your cluster's version
- Partition: Add `#SBATCH --partition=your-partition` if required
- Account: Add `#SBATCH --account=your-account` if required

## Directory Structure After Setup

```
/om2/user/mabdel03/
├── anaconda/                           # Your anaconda installation
│   └── etc/profile.d/conda.sh         # Conda initialization script
│
├── conda_envs/
│   └── SLT_Proj_Env/                  # Project conda environment
│       ├── bin/
│       ├── lib/
│       └── ...
│
└── 9.520-The-Geometry-of-Grok/        # This repository
    ├── setup_conda_env.sh             # Environment setup script
    ├── Replications/                   # Paper replications
    └── New_Explorations/               # GOP analysis framework
```

## Storage Planning

Before running full experiments, check available storage:

```bash
# Check your quota
quota -s

# Check available space
df -h /om2/user/mabdel03
```

### Storage Requirements

| Experiment | Description | Total Storage |
|-----------|-------------|---------------|
| 09 Levi | Linear (1K params) | ~16 GB ⭐ GOOD FOR TESTING |
| 08 Doshi | Poly (50K params) | ~50 GB |
| 03 Nanda | Transformer (100K) | ~160 GB |
| 07 Thilak | Slingshot (150K) | ~450 GB |
| 06 Humayun | MNIST MLP (160K) | ~1 TB |

**Tip:** Run smaller experiments first to verify everything works, then scale up based on your storage quota.

## Running Multiple Experiments

### Sequential Submission

```bash
cd New_Explorations/experiments

# Submit jobs one by one
sbatch 09_levi_linear/run_gop_analysis.sh
sbatch 08_doshi_polynomials/run_gop_analysis.sh
sbatch 03_nanda_progress/run_gop_analysis.sh
```

### Parallel Submission (if you have resources)

```bash
for dir in 09_levi_linear 08_doshi_polynomials 03_nanda_progress; do
    cd $dir
    sbatch run_gop_analysis.sh
    cd ..
done
```

### Using Job Dependencies

```bash
# Run in sequence using dependencies
JOB1=$(sbatch --parsable 09_levi_linear/run_gop_analysis.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 08_doshi_polynomials/run_gop_analysis.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 03_nanda_progress/run_gop_analysis.sh)

echo "Submitted jobs: $JOB1, $JOB2, $JOB3"
```

## Monitoring Jobs

```bash
# Check queue
squeue -u mabdel03

# Check specific job
squeue -j JOBID

# Cancel job if needed
scancel JOBID

# View job details
scontrol show job JOBID

# Check completed jobs
sacct -u mabdel03 --format=JobID,JobName,Partition,State,ExitCode,Elapsed
```

## Updating Repository

If the code is updated on GitHub:

```bash
cd /om2/user/mabdel03/9.520-The-Geometry-of-Grok
git pull origin main
```

The conda environment persists, so no need to reinstall unless dependencies change.

## Troubleshooting

### Conda Not Found

```bash
# Make sure anaconda is in the correct location
ls /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh

# If not, find your conda installation
which conda
# Then update the path in setup_conda_env.sh and all SLURM scripts
```

### GPU Not Available in Job

```bash
# Check available GPU partitions
sinfo -o "%20P %5a %10l %6D %6t %N"

# Your job might need a specific partition
#SBATCH --partition=gpu-partition-name
```

### Out of Storage

```bash
# Check usage
du -sh New_Explorations/results/*

# Delete old results if needed
rm -rf New_Explorations/results/old_experiment/

# Or use metrics-only mode (edit config.yaml):
# storage:
#   store_full_gop: false
```

### Module Load Fails

```bash
# Check available modules
module avail

# Find CUDA version
module avail cuda

# Update SLURM scripts with correct module names
```

## Environment Variables (Optional)

Add to your `~/.bashrc`:

```bash
# Grokking project shortcuts
export GROK_HOME="/om2/user/mabdel03/9.520-The-Geometry-of-Grok"
export GROK_ENV="/om2/user/mabdel03/conda_envs/SLT_Proj_Env"

# Conda setup
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh

# Quick activation alias
alias grok_env="conda activate $GROK_ENV"
alias grok_cd="cd $GROK_HOME"

# Then you can simply type:
# grok_cd && grok_env
```

## Verifying Successful Setup

Run this checklist:

```bash
# 1. Conda environment exists
conda env list | grep SLT_Proj_Env
# Should show: /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# 2. PyTorch with CUDA works
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env
python -c "import torch; assert torch.cuda.is_available(); print('✓ CUDA available')"

# 3. All packages installed
python -c "import h5py, yaml, scipy, matplotlib; print('✓ All packages installed')"

# 4. Framework tests pass
cd New_Explorations
python test_framework.py
# Should end with: ✅ END-TO-END TEST PASSED!

# 5. Can submit jobs
cd experiments/09_levi_linear
sbatch --test-only run_gop_analysis.sh
# Should show: sbatch: Job XXXXX to start at ...
```

If all checks pass: **✅ Setup is complete and ready to use!**

## Next Steps

1. **Read the documentation:**
   - `README.md` - Project overview
   - `New_Explorations/START_HERE.md` - Quick start guide
   - `New_Explorations/GETTING_STARTED.md` - Detailed tutorial

2. **Run a test experiment:**
   - Start with `09_levi_linear` (smallest, fastest)
   - Verify outputs are created
   - Check visualizations work

3. **Scale up:**
   - Run progressively larger experiments
   - Monitor storage usage
   - Analyze GOP dynamics

## Support

If you encounter issues:
1. Check logs in `experiments/XX/logs/`
2. Verify conda environment is activated
3. Ensure SLURM parameters match your cluster
4. Check storage quota: `quota -s`
5. Review error messages in `.err` files

---

**Happy grokking research! 🎓**

