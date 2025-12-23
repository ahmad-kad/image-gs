#!/bin/bash
# Image-GS Docker Runner Script with GPU Support
# This script runs Image-GS using the custom GPU-enabled Docker image

echo "Running Image-GS with GPU support..."

docker run --rm \
  -e PYTHONUNBUFFERED=1 \
  --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  image-gs-image-gs \
  bash -c "
    echo 'Installing fused-ssim (runtime only)...'
    pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

    echo 'Running Image-GS with GPU acceleration...'
    python main.py --input_path='images/anime-1_2k.png' --exp_name='test/anime-1_2k-gpu' --num_gaussians=10000 --quantize
  "

echo "Image-GS GPU training completed successfully!"