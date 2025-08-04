# Use the official PyTorch image with CUDA support (or CPU-only version)
# For GPU support, use: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
# For CPU-only, use: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime or pytorch/pytorch:latest
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

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
    seaborn \
    plotly \
    scikit-learn \
    "opencv-python-headless>=4.5.0,<4.10.0" \
    Pillow \
    tqdm \
    tensorboard \
    torchvision \
    torchaudio \
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
RUN mkdir -p /workspace/notebooks /workspace/data /workspace/models /workspace/scripts

# Set up Jupyter configuration
RUN jupyter lab --generate-config

# Expose ports
EXPOSE 8888 6006

# Set the default command
CMD ["bash"]
