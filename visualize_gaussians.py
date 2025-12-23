#!/usr/bin/env python3
"""
Script to visualize Gaussians from a trained checkpoint
"""
import torch
import numpy as np
import os
from pathlib import Path

# Import project utilities
from utils.image_utils import visualize_gaussian_position, visualize_gaussian_footprint, load_images

def visualize_checkpoint_gaussians(checkpoint_path, exp_name="visualized_checkpoint"):
    """
    Load checkpoint and create visualizations of the Gaussians
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint loaded successfully")
    print("Available keys:", list(checkpoint.keys()))

    # Extract model parameters
    model_state = checkpoint['state_dict']

    # Get Gaussian parameters
    xy = model_state['xy']          # positions (N, 2)
    scale = model_state['scale']     # scales (N, 2)
    rot = model_state['rot']         # rotations (N, 1)
    feat = model_state['feat']       # features/colors (N, C)

    print(f"Found {xy.shape[0]} Gaussians")
    print(f"Feature dimensions: {feat.shape[1]}")

    # Determine input channels from feature dimensions
    # Assume RGB (3 channels) for visualization
    input_channels = [3] if feat.shape[1] >= 3 else [feat.shape[1]]

    # Create output directory
    output_dir = Path(f"results/{exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load original image for visualization
    # Look for ground truth image in the results
    gt_image_path = None
    for root, dirs, files in os.walk("results"):
        for file in files:
            if file.startswith("gt_res-") and file.endswith(".jpg"):
                gt_image_path = os.path.join(root, file)
                break
        if gt_image_path:
            break

    if gt_image_path:
        print(f"Loading ground truth image: {gt_image_path}")
        gt_images, input_channels, _, _ = load_images(gt_image_path, downsample_ratio=None)
        # gt_images is in CHW format, convert to what visualization expects
        images = gt_images  # This is already in the right format for visualization
    else:
        print("Ground truth image not found, creating blank background")
        # Create a blank white image for visualization (RGB)
        img_size = (3, 2048, 2048)  # CHW format: channels, height, width
        blank_image = np.ones(img_size, dtype=np.float32)
        images = blank_image
        input_channels = [3]

    # Visualize Gaussian positions (every 20th Gaussian to avoid clutter)
    print("Creating Gaussian position visualization...")
    pos_viz_path = output_dir / "gaussian_positions.jpg"
    visualize_gaussian_position(
        str(pos_viz_path),
        images=images,
        xy=xy,
        input_channels=input_channels,
        color="#7bf1a8",  # Light green
        size=500,         # Dot size
        every_n=20,       # Show every 20th Gaussian
        alpha=0.7
    )
    print(f"Saved: {pos_viz_path}")

    # Visualize Gaussian footprints (shapes and colors)
    print("Creating Gaussian footprint visualization...")
    footprint_viz_path = output_dir / "gaussian_footprints.jpg"
    visualize_gaussian_footprint(
        str(footprint_viz_path),
        xy=xy,
        scale=scale,
        rot=rot,
        feat=feat,
        img_h=2048,
        img_w=2048,
        input_channels=input_channels,
        alpha=0.8
    )
    print(f"Saved: {footprint_viz_path}")

    # Create a subset visualization with more Gaussians for detail
    print("Creating detailed Gaussian position visualization (every 5th)...")
    detailed_pos_viz_path = output_dir / "gaussian_positions_detailed.jpg"
    visualize_gaussian_position(
        str(detailed_pos_viz_path),
        images=images,
        xy=xy,
        input_channels=input_channels,
        color="#ff6b6b",  # Red for detail view
        size=200,         # Smaller dots
        every_n=5,        # Show every 5th Gaussian
        alpha=0.5
    )
    print(f"Saved: {detailed_pos_viz_path}")

    print(f"\nVisualization complete! Check: {output_dir}")
    print("Files created:")
    print(f"  - {pos_viz_path.name}: Gaussian positions (sparse)")
    print(f"  - {detailed_pos_viz_path.name}: Gaussian positions (detailed)")
    print(f"  - {footprint_viz_path.name}: Gaussian footprints (shapes & colors)")

if __name__ == "__main__":
    # Default checkpoint path - adjust as needed
    checkpoint_path = "results/test/anime-1_2k-gpu/num-10000_inv-scale-5.0_bits-16-16-16-16_top-10_g-0.3_l1-1.0_l2-0.0_ssim-0.1_decay-1-10.0_prog/checkpoints/ckpt_step-5300.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable to point to your checkpoint file")
        exit(1)

    exp_name = "visualized_gaussians"
    visualize_checkpoint_gaussians(checkpoint_path, exp_name)
