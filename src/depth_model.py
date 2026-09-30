"""Depth Estimation Module (Task Card T2).

Single-View Monocular Height & Relative Depth Estimation.
Implements the core backbone interface using Depth-Anything-V2.

Inputs:
    - 2D Optical Image: RGB image (PNG, JPG, or GeoTIFF) as file path, PIL Image, or np.ndarray.
Outputs:
    - Relative DSM (rDSM): Normalized float32 numpy array in range [0.0, 1.0],
      where 1.0 = highest structure / elevation, and 0.0 = lowest ground / depression.
    - Export artifacts: .npy array, 16-bit PNG heightmap, 8-bit visual displacement map.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "depth-anything/Depth-Anything-V2-Small-hf",
        "device": "auto",
    },
    "calibration": {
        "water_mask": {
            "enabled": True,
            "min_blue_diff": 35,
            "max_red": 120,
        }
    },
}


def load_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Loads configuration YAML file with robust fallback defaults.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    path = Path(config_path)
    if path.exists() and yaml is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    return loaded
        except Exception:
            pass
    return DEFAULT_CONFIG


class DepthEstimator:
    """Monocular depth estimator using pre-trained vision transformer backbones.

    Provides inference, normalization to [0.0, 1.0] relative DSM (rDSM),
    and automated water-body depression masking.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        config_path: Union[str, Path] = "config.yaml",
    ) -> None:
        """Initializes the depth estimator.

        Args:
            model_name: Hugging Face model identifier. If None, loaded from config.yaml.
            device: 'cuda', 'cpu', or 'auto'. If None, loaded from config.yaml.
            config_path: Path to configuration YAML file.
        """
        self.config = load_config(config_path)
        self.model_name = model_name or self.config.get("model", {}).get(
            "name", DEFAULT_CONFIG["model"]["name"]
        )
        configured_device = device or self.config.get("model", {}).get("device", "auto")

        if configured_device == "auto":
            try:
                import torch

                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = configured_device

        self._pipeline = None

    @property
    def pipeline(self) -> Any:
        """Lazily instantiates and returns the Hugging Face depth pipeline."""
        if self._pipeline is None:
            from transformers import pipeline

            print(f"[DepthEstimator] Loading model '{self.model_name}' on device '{self.device}'...", flush=True)
            try:
                # Fast path when model weights are cached offline
                self._pipeline = pipeline(
                    "depth-estimation",
                    model=self.model_name,
                    device=self.device,
                    model_kwargs={"local_files_only": True},
                )
            except Exception:
                # Fallback to standard online resolution
                self._pipeline = pipeline(
                    "depth-estimation",
                    model=self.model_name,
                    device=self.device,
                )
            print("[DepthEstimator] Model loaded successfully.", flush=True)
        return self._pipeline

    def detect_water_mask(
        self,
        rgb_image: Image.Image,
        min_blue_diff: int = 35,
        max_red: int = 120,
    ) -> np.ndarray:
        """Detects water bodies using chromatic thresholding to prevent false elevation mounds.

        Args:
            rgb_image: Input PIL RGB image.
            min_blue_diff: Required excess of blue channel over red & green.
            max_red: Maximum red intensity threshold for dark aquatic bodies.

        Returns:
            np.ndarray: Boolean mask of shape (H, W), True where water is detected.
        """
        arr = np.asarray(rgb_image, dtype=np.int16)
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        water_mask = (b > (r + min_blue_diff)) & (b > (g + 10)) & (r < max_red)
        return water_mask

    def predict_relative_depth(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        apply_water_mask: bool = True,
    ) -> Tuple[np.ndarray, Image.Image]:
        """Predicts normalized relative depth map (rDSM) from an input optical image.

        Args:
            image: File path (str or Path), PIL Image, or RGB NumPy array of shape (H, W, 3).
            apply_water_mask: Whether to depress water bodies to base ground level.

        Returns:
            Tuple[np.ndarray, Image.Image]:
                - depth_norm: Float32 array of shape (H, W) in range [0.0, 1.0],
                  where 1.0 is highest/tallest structure and 0.0 is lowest ground.
                - pil_img: The standardized RGB PIL Image instance.
        """
        # 1. Normalize input to PIL RGB Image
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 2. Run model inference
        pipe = self.pipeline
        result = pipe(pil_img)
        pred_depth = result["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)

        # 3. Normalize to [0.0, 1.0] range
        d_min = float(pred_depth.min())
        d_max = float(pred_depth.max())
        denom = d_max - d_min if (d_max - d_min) > 1e-8 else 1e-8
        depth_norm = (pred_depth - d_min) / denom

        # Resize predicted depth array to match original image pixel dimensions if needed
        w, h = pil_img.size
        if depth_norm.shape != (h, w):
            depth_img = Image.fromarray(depth_norm).resize((w, h), Image.Resampling.BICUBIC)
            depth_norm = np.asarray(depth_img, dtype=np.float32)

        # 4. Apply water depression mask (Task T15)
        water_cfg = self.config.get("calibration", {}).get("water_mask", {})
        if apply_water_mask and water_cfg.get("enabled", True):
            min_blue = water_cfg.get("min_blue_diff", 35)
            max_r = water_cfg.get("max_red", 120)
            water_mask = self.detect_water_mask(pil_img, min_blue_diff=min_blue, max_red=max_r)
            if np.any(water_mask):
                depth_norm[water_mask] = np.minimum(depth_norm[water_mask], 0.01)

        # Re-clamp to strictly [0.0, 1.0]
        depth_norm = np.clip(depth_norm, 0.0, 1.0).astype(np.float32)
        return depth_norm, pil_img

    def save_depth_outputs(
        self,
        depth: np.ndarray,
        output_prefix: Union[str, Path],
        original_image: Optional[Image.Image] = None,
    ) -> Dict[str, Path]:
        """Saves relative depth outputs across standard formats (.npy, 16-bit PNG, 8-bit PNG).

        Args:
            depth: Float32 array of shape (H, W) in range [0.0, 1.0].
            output_prefix: Target file path prefix (without extension). Parent dirs created automatically.
            original_image: Optional PIL Image to save alongside as paired RGB texture.

        Returns:
            Dict[str, Path]: Paths to the generated artifact files.
        """
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, Path] = {}

        # 1. Raw relative DSM (.npy float32 array)
        npy_path = prefix.with_name(f"{prefix.name}_rdsm.npy")
        np.save(npy_path, depth.astype(np.float32))
        saved_files["npy"] = npy_path

        # 2. 16-bit high-precision heightmap PNG (values 0..65535)
        png_16bit_path = prefix.with_name(f"{prefix.name}_depth_16bit.png")
        depth_16bit = (depth * 65535.0).astype(np.uint16)
        Image.fromarray(depth_16bit).save(png_16bit_path)
        saved_files["depth_16bit"] = png_16bit_path

        # 3. 8-bit standard displacement map PNG (values 0..255)
        png_8bit_path = prefix.with_name(f"{prefix.name}_depth.png")
        depth_8bit = (depth * 255.0).astype(np.uint8)
        Image.fromarray(depth_8bit, mode="L").save(png_8bit_path)
        saved_files["depth_8bit"] = png_8bit_path

        # 4. Paired RGB texture
        if original_image is not None:
            rgb_path = prefix.with_name(f"{prefix.name}_rgb.png")
            original_image.save(rgb_path)
            saved_files["rgb"] = rgb_path

        return saved_files


# Convenient functional entry point for direct imports
_default_estimator: Optional[DepthEstimator] = None


def predict_relative_depth(
    image: Union[str, Path, Image.Image, np.ndarray],
    apply_water_mask: bool = True,
    config_path: Union[str, Path] = "config.yaml",
) -> Tuple[np.ndarray, Image.Image]:
    """Module-level function for predicting relative depth.

    Args:
        image: File path, PIL Image, or RGB NumPy array.
        apply_water_mask: Whether to depress water bodies to base ground level.
        config_path: Path to configuration YAML.

    Returns:
        Tuple[np.ndarray, Image.Image]: (depth_norm [0.0..1.0], pil_image).
    """
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = DepthEstimator(config_path=config_path)
    return _default_estimator.predict_relative_depth(image, apply_water_mask=apply_water_mask)


def main() -> None:
    """CLI runner for standalone inference tests."""
    parser = argparse.ArgumentParser(
        description="DepthWizard Task T2: Predict relative depth map from an aerial/satellite image."
    )
    parser.add_argument("image_path", type=str, help="Path to input 2D image (PNG/JPG/GeoTIFF).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs",
        help="Directory to save output files (default: data/outputs).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--no-water-mask",
        action="store_true",
        help="Disable automatic water depression masking.",
    )

    args = parser.parse_args()
    input_path = Path(args.image_path)
    if not input_path.exists():
        print(f"Error: Input image '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    estimator = DepthEstimator(config_path=args.config)
    depth, pil_img = estimator.predict_relative_depth(
        input_path, apply_water_mask=not args.no_water_mask
    )

    out_dir = Path(args.output_dir)
    stem = input_path.stem
    out_prefix = out_dir / stem

    artifacts = estimator.save_depth_outputs(depth, out_prefix, original_image=pil_img)

    print("\n--- Depth Estimation Completed ---", flush=True)
    print(f"Input image shape:   {pil_img.size[0]}x{pil_img.size[1]} px", flush=True)
    print(f"Output rDSM shape:   {depth.shape}", flush=True)
    print(f"Relative range:      min={depth.min():.4f}, max={depth.max():.4f}", flush=True)
    print("Generated Artifacts:", flush=True)
    for key, path in artifacts.items():
        print(f"  [{key}]: {path}", flush=True)


if __name__ == "__main__":
    main()
