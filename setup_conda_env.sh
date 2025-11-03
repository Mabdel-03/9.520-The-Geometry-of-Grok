#!/bin/bash
#
# Setup Script for Conda Environment on HPC Cluster
# Creates and configures the SLT_Proj_Env environment at /om2/user/mabdel03/conda_envs/SLT_Proj_Env
#

set -e  # Exit on error

echo "=========================================="
echo "Setting up Conda Environment for Grokking Project"
echo "=========================================="

# Source conda
echo "Sourcing conda..."
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh

# Create conda environment
CONDA_ENV_PATH="/om2/user/mabdel03/conda_envs/SLT_Proj_Env"

if conda env list | grep -q "$CONDA_ENV_PATH"; then
    echo "Environment already exists at $CONDA_ENV_PATH"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -p $CONDA_ENV_PATH -y
    else
        echo "Skipping environment creation."
        echo "To activate: source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh && conda activate $CONDA_ENV_PATH"
        exit 0
    fi
fi

echo "Creating conda environment..."
conda create -p $CONDA_ENV_PATH python=3.9 -y

# Activate environment
echo "Activating environment..."
conda activate $CONDA_ENV_PATH

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core scientific computing packages
echo "Installing scientific computing packages..."
pip install numpy scipy matplotlib pandas scikit-learn

# Install HDF5 and data handling
echo "Installing data handling packages..."
pip install h5py pyyaml tqdm psutil

# Install additional useful packages
echo "Installing additional packages..."
pip install jupyter ipython tensorboard

# Install transformers and NLP packages (for Paper 4)
echo "Installing transformers and NLP packages..."
pip install transformers datasets tokenizers

# Install graph/molecular packages (for Papers 5)
echo "Installing specialized packages..."
pip install networkx  # For graph datasets

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python -c "import h5py; print(f'h5py: {h5py.__version__}')"
python -c "import yaml; print(f'PyYAML: {yaml.__version__}')"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Environment location: $CONDA_ENV_PATH"
echo ""
echo "To activate the environment in future sessions:"
echo "  source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh"
echo "  conda activate $CONDA_ENV_PATH"
echo ""
echo "To use in SLURM scripts, add these lines:"
echo "  source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh"
echo "  conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env"
echo ""
echo "All SLURM scripts in this repo are already configured to use this environment!"
echo ""

