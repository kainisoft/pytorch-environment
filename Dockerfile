# Use Python base image and install PyTorch manually for Apple Silicon compatibility
# This ensures we get ARM64 native performance and MPS support
FROM python:3.11-slim

# Set the working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with MPS support for Apple Silicon
# Using the nightly build which has better Apple Silicon support
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    torchmetrics

# Install Python packages for data science and machine learning
# Note: NumPy version pinned to <2.0 for PyTorch 2.1.0 compatibility
# OpenCV version constrained to avoid NumPy 2.x compatibility issues
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    "numpy<2.0" \
    pandas \
    matplotlib \
    mlxtend \
    seaborn \
    plotly \
    scikit-learn \
    "opencv-python-headless>=4.5.0,<4.10.0" \
    Pillow \
    tqdm \
    tensorboard \
    transformers \
    datasets \
    accelerate \
    wandb \
    mlflow \
    optuna \
    albumentations \
    timm

# Install additional useful packages
RUN pip install --no-cache-dir \
    requests \
    beautifulsoup4 \
    lxml \
    h5py \
    scipy \
    statsmodels \
    networkx \
    bokeh \
    streamlit

# Create directories for mounting volumes
RUN mkdir -p /workspace/features

# Set up Jupyter configuration
RUN jupyter lab --generate-config

# Expose ports
EXPOSE 8888 6006

# Set the default command
CMD ["bash"]
