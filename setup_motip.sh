#!/bin/bash
set -e  # Exit on any error

echo "Setting up MOTIP environment..."

# Remove old environment if it exists
echo "Removing old environment if it exists..."
conda env remove -n MOTIP -y 2>/dev/null || true

# Create new environment with all dependencies resolved together
echo "Creating conda environment with dependencies..."
conda create -n MOTIP python=3.12 pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 pyyaml tqdm matplotlib scipy pandas -c pytorch -c nvidia -y

# Activate environment
echo "Activating environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MOTIP

# Install pip-only packages
echo "Installing pip packages..."
pip install wandb accelerate einops idna

# Install CUDA toolkit and compatible GCC
echo "Installing CUDA toolkit 12.1..."
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y

echo "Installing GCC 11.2..."
conda install gxx_linux-64=11.2.0 gcc_linux-64=11.2.0 -y

# Uninstall conda PyTorch and reinstall via pip (to avoid Intel library conflict)
echo "Reinstalling PyTorch via pip..."
conda uninstall pytorch torchvision torchaudio pytorch-cuda -y
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Fix broken libcudart symlink
echo "Fixing libcudart symlink..."
cd $CONDA_PREFIX/lib
ln -sf libcudart.so.12.1.105 libcudart.so

# Set environment variables and compile Deformable Attention
echo "Compiling Deformable Attention extension..."
cd ~/Desktop/11-MOTIP/models/ops/
rm -rf build
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.9"
python setup.py build install

# Test the installation
echo "Testing installation..."
python test.py

echo "Setup complete! Environment MOTIP is ready."
echo "To activate: conda activate MOTIP"
