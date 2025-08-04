# PyTorch Learning Environment with Docker Compose

This repository provides a complete PyTorch development environment using Docker Compose, perfect for learning and experimenting with PyTorch without installing it directly on your host system.

## Features

- **PyTorch 2.1.0** with CUDA support (configurable for CPU-only)
- **Jupyter Lab** for interactive development
- **Pre-installed packages** for data science and machine learning
- **Volume mounting** for persistent data and code
- **Port mapping** for easy access to Jupyter and TensorBoard
- **GPU support** (optional, requires NVIDIA Docker)

## Quick Start

### 1. Build and Start the Environment

```bash
# Build the Docker image and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

### 2. Access Jupyter Lab

Once the container is running, open your browser and navigate to:

```
http://localhost:8888
```

**Login token:** `pytorch-learning`

### 3. Stop the Environment

```bash
# Stop the container
docker-compose down

# Stop and remove volumes (WARNING: This will delete your data)
docker-compose down -v
```

## Directory Structure

The following directories will be created and mounted as volumes:

```
pet7/
├── docker-compose.yml
├── Dockerfile
├── README.md
├── notebooks/          # Your Jupyter notebooks (persistent)
├── data/              # Dataset storage (persistent)
├── models/            # Trained model storage (persistent)
└── scripts/           # Python scripts (persistent)
```

## Installed Packages

**Note:** Package versions are carefully managed for compatibility:
- **NumPy** is pinned to `<2.0` to ensure compatibility with PyTorch 2.1.0
- **OpenCV** is constrained to `>=4.5.0,<4.10.0` to avoid NumPy 2.x conflicts

### Core PyTorch Stack
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `torchaudio` - Audio processing utilities

### Data Science & Visualization
- `numpy`, `pandas` - Data manipulation
- `matplotlib`, `seaborn`, `plotly` - Data visualization
- `scikit-learn` - Traditional machine learning
- `scipy`, `statsmodels` - Scientific computing

### Computer Vision & Image Processing
- `opencv-python-headless` - Computer vision
- `Pillow` - Image processing
- `albumentations` - Image augmentation
- `timm` - Pre-trained vision models

### NLP & Transformers
- `transformers` - Hugging Face transformers
- `datasets` - Hugging Face datasets
- `accelerate` - Distributed training

### Experiment Tracking & Optimization
- `tensorboard` - Experiment visualization
- `wandb` - Weights & Biases integration
- `mlflow` - ML lifecycle management
- `optuna` - Hyperparameter optimization

### Development Tools
- `jupyter`, `jupyterlab` - Interactive development
- `tqdm` - Progress bars
- `requests`, `beautifulsoup4` - Web scraping

## GPU Support

To enable GPU support:

1. Install [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
2. Uncomment the GPU configuration in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. Rebuild the container:

```bash
docker-compose up --build
```

## Customization

### Change PyTorch Version

Edit the `FROM` line in `Dockerfile`:

```dockerfile
# For different PyTorch versions:
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime  # Older version
FROM pytorch/pytorch:latest                           # Latest version
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel    # Development version
```

### Add More Packages

Add packages to the `RUN pip install` commands in `Dockerfile`:

```dockerfile
RUN pip install --no-cache-dir \
    your-package-name \
    another-package
```

### Change Jupyter Token

Modify the `JUPYTER_TOKEN` environment variable in `docker-compose.yml`:

```yaml
environment:
  - JUPYTER_TOKEN=your-custom-token
```

## Common Commands

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs

# Execute commands in the running container
docker-compose exec pytorch-jupyter bash

# Rebuild after changes
docker-compose up --build

# Pull latest PyTorch image
docker-compose pull
```

## Troubleshooting

### Port Already in Use
If port 8888 is already in use, change it in `docker-compose.yml`:

```yaml
ports:
  - "8889:8888"  # Use port 8889 instead
```

### Permission Issues
If you encounter permission issues with mounted volumes:

```bash
# Fix ownership (Linux/macOS)
sudo chown -R $USER:$USER notebooks/ data/ models/ scripts/
```

### GPU Not Detected
Verify GPU support:

```bash
# Inside the container
python -c "import torch; print(torch.cuda.is_available())"
```

### NumPy Compatibility Issues
If you encounter NumPy compatibility errors with PyTorch imports:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.0...
```

**Solution:** This is already fixed in the current Dockerfile, but if you encounter it:

```bash
# Inside the container
pip install "numpy<2.0"
pip install "opencv-python-headless>=4.5.0,<4.10.0"
```

**Root Cause:** PyTorch 2.1.0 was compiled against NumPy 1.x and is incompatible with NumPy 2.x.

## Learning Resources

With this environment set up, you can start with:

1. [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
2. [Deep Learning with PyTorch Book](https://pytorch.org/deep-learning-with-pytorch)
3. [PyTorch Examples Repository](https://github.com/pytorch/examples)

Happy learning! 🚀
