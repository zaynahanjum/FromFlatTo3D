#!/usr/bin/env python3
"""
make_depth.py - Day-0 MVP Depth Inference Script
Runs Depth Anything V2 Small on an optical satellite/aerial image
and saves paired RGB and normalized depth maps for the 3D viewer.
Includes automated water body masking (Task T15) to guarantee water sinks below terrain.
"""

import sys
import os
import argparse
import numpy as np
from PIL import Image, ImageFilter
import torch
from transformers import pipeline


def apply_water_mask(rgb_img: Image.Image, depth_arr: np.ndarray) -> np.ndarray:
    """
    Detects water bodies (blue-dominant areas) and clamps their height
    so that water sits as a realistic flat depression beneath the surrounding land.
    """
    rgb_np = np.array(rgb_img, dtype=np.float32)
    r = rgb_np[:, :, 0]
    g = rgb_np[:, :, 1]
    b = rgb_np[:, :, 2]

    # Water criterion: blue-dominant, saturated water pixels (must have low red, unlike gray/white rooftops)
    water_mask = (b > r + 35) & (b > g + 10) & (b > 50) & (r < 120)

    if np.any(water_mask):
        print(f"[*] Detected water body pixels ({np.sum(water_mask)} pixels). Applying water basin depression...")
        # Create a smoothed boundary mask for natural lake shorelines
        mask_img = Image.fromarray((water_mask * 255).astype(np.uint8))
        blurred_mask = np.array(mask_img.filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32) / 255.0

        # Find surrounding ground level near the water
        surrounding_ground = float(np.percentile(depth_arr[~water_mask], 10))
        water_level = max(0.01, surrounding_ground - 0.08)

        # Blend water level into the depth map
        depth_arr = (1.0 - blurred_mask) * depth_arr + blurred_mask * water_level

    return depth_arr


def estimate_depth(image_path: str, output_dir: str = "viewer/public", invert: bool = False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    device = 0 if torch.cuda.is_available() else -1
    device_name = "CUDA (GPU)" if device == 0 else "CPU"
    print(f"[*] Running depth estimation on: {device_name}")

    print("[*] Loading Depth Anything V2 Small model...")
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

    print(f"[*] Loading input image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    print("[*] Running inference...")
    result = pipe(img)
    pred_depth = result["predicted_depth"]

    if isinstance(pred_depth, torch.Tensor):
        d = pred_depth.squeeze().cpu().numpy()
    else:
        d = np.array(pred_depth).squeeze()

    # Normalize depth to [0, 1]
    d_min, d_max = d.min(), d.max()
    print(f"[*] Raw depth range: min={d_min:.4f}, max={d_max:.4f}")
    if d_max > d_min:
        d_norm = (d - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros_like(d)

    if invert:
        print("[*] Inverting depth map (invert=True)...")
        d_norm = 1.0 - d_norm

    # Resize depth to original image dimensions
    depth_resized = np.array(
        Image.fromarray((d_norm * 255.0).astype(np.uint8)).resize(original_size, resample=Image.Resampling.BILINEAR),
        dtype=np.float32
    ) / 255.0

    # Apply automated water masking (Task T15)
    depth_fixed = apply_water_mask(img, depth_resized)

    depth_uint8 = np.clip(depth_fixed * 255.0, 0, 255).astype(np.uint8)
    final_depth_img = Image.fromarray(depth_uint8)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_rgb_path = os.path.join(output_dir, f"{base_name}_rgb.png")
    out_depth_path = os.path.join(output_dir, f"{base_name}_depth.png")

    img.save(out_rgb_path)
    final_depth_img.save(out_depth_path)

    print(f"[+] Saved RGB texture: {out_rgb_path}")
    print(f"[+] Saved Depth map:   {out_depth_path}")
    return out_rgb_path, out_depth_path


def main():
    parser = argparse.ArgumentParser(description="Estimate relative depth from satellite/aerial image.")
    parser.add_argument("image", help="Path to input satellite/aerial image (PNG/JPG)")
    parser.add_argument("--output-dir", default="viewer/public", help="Directory to save viewer assets")
    parser.add_argument("--invert", action="store_true", help="Invert depth polarity")
    args = parser.parse_args()

    estimate_depth(args.image, args.output_dir, args.invert)


if __name__ == "__main__":
    main()
