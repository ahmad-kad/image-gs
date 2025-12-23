# Use PyTorch CUDA base image
FROM pytorch/pytorch:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Copy the entire codebase first
COPY . .

# Install additional packages in the existing PyTorch environment
RUN pip install lpips==0.1.4 matplotlib==3.9.2 numpy==2.0.2 opencv-python==4.12.0.88 pytorch-msssim==1.0.0 scikit-image==0.24.0 scipy==1.13.1 torchmetrics==1.5.2 flip-evaluator && \
    cd gsplat && \
    pip install -e ".[dev]" && \
    cd ..

# Set the default command
CMD ["/bin/bash"]
