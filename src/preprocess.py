"""Image Preprocessing, Geospatial Detection & Tiling Module (Task Card T3).

Handles optical imagery (PNG, JPG, or GeoTIFF) for single-view depth estimation:
1. Inspects image metadata and detects georeferencing (CRS, affine transform, bounding box, resolution).
2. Decomposes large satellite scenes into overlapping sliding-window tiles.
3. Provides 2D feathering blend stitching to eliminate hard seams in reconstructed DSMs.

Inputs:
    - Optical Image: Path (str/Path), PIL Image, or RGB NumPy array of shape (H, W, 3).
Outputs:
    - Standardized RGB NumPy array of shape (H, W, 3) in uint8 [0..255].
    - GeoMetadata: Coordinate reference system, bounding box, pixel resolution, or None if unreferenced.
    - Overlapping tiles and seamless blend-stitching routines.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import yaml
except ImportError:
    yaml = None

# Optional geospatial backend
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


DEFAULT_TILING_CONFIG: Dict[str, Any] = {
    "tile_size": 512,
    "overlap": 64,
    "blend_method": "linear",
}


@dataclass
class GeoMetadata:
    """Geospatial metadata extracted from georeferenced raster files."""

    is_georeferenced: bool
    crs: Optional[str] = None
    transform: Optional[Tuple[float, ...]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (left, bottom, right, top) in native CRS
    bounds_latlon: Optional[Tuple[float, float, float, float]] = None  # (min_lon, min_lat, max_lon, max_lat)
    resolution: Optional[Tuple[float, float]] = None  # (dx, dy) in native units per pixel
    width: int = 0
    height: int = 0
    channels: int = 3
    nodata: Optional[float] = None
    driver: Optional[str] = None

    def summary(self) -> str:
        """Returns a formatted human-readable summary of geographic attributes."""
        if not self.is_georeferenced:
            return (
                f"Non-georeferenced image ({self.width}x{self.height} px, {self.channels} channels)."
            )
        crs_str = self.crs or "Unknown"
        res_str = (
            f"{self.resolution[0]:.4f} x {self.resolution[1]:.4f}"
            if self.resolution
            else "Unknown"
        )
        bounds_str = (
            f"[{self.bounds[0]:.2f}, {self.bounds[1]:.2f}, {self.bounds[2]:.2f}, {self.bounds[3]:.2f}]"
            if self.bounds
            else "None"
        )
        return (
            f"Georeferenced GeoTIFF:\n"
            f"  - Dimensions:  {self.width}x{self.height} px ({self.channels} channels)\n"
            f"  - CRS:         {crs_str}\n"
            f"  - Resolution:  {res_str} units/px\n"
            f"  - Bounds:      {bounds_str}\n"
            f"  - Driver:      {self.driver or 'RasterIO'}"
        )


@dataclass
class TileCrop:
    """An individual overlapping tile crop with its source bounding coordinates."""

    crop: np.ndarray  # Array of shape (tile_h, tile_w, C) or (tile_h, tile_w)
    bbox: Tuple[int, int, int, int]  # (y0, y1, x0, x1) in full image coordinates
    tile_index: int


def load_tiling_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Loads tiling configuration parameters from config.yaml."""
    cfg = DEFAULT_TILING_CONFIG.copy()
    path = Path(config_path)
    if path.exists() and yaml is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "tiling" in data:
                    cfg.update(data["tiling"])
        except Exception:
            pass
    return cfg


def inspect_image(
    image_input: Union[str, Path, Image.Image, np.ndarray]
) -> Tuple[np.ndarray, GeoMetadata]:
    """Inspects any image, extracts geospatial metadata if available, and returns RGB array.

    Args:
        image_input: Path to file, PIL Image, or NumPy array.

    Returns:
        Tuple[np.ndarray, GeoMetadata]:
            - rgb_array: NumPy uint8 array of shape (H, W, 3).
            - metadata: GeoMetadata describing spatial referencing and dimensions.
    """
    # Case 1: Georeferenced file on disk (via rasterio if available)
    if isinstance(image_input, (str, Path)):
        file_path = Path(image_input)
        if not file_path.exists():
            raise FileNotFoundError(f"Image not found at '{file_path}'")

        if HAS_RASTERIO:
            try:
                with rasterio.open(file_path) as src:
                    crs = src.crs.to_string() if src.crs else None
                    is_geo = crs is not None

                    # Read RGB bands
                    count = src.count
                    if count >= 3:
                        bands = [src.read(i) for i in (1, 2, 3)]
                        rgb = np.stack(bands, axis=-1)
                    else:
                        single = src.read(1)
                        rgb = np.stack([single, single, single], axis=-1)

                    # Normalize to uint8 [0..255] if necessary
                    if rgb.dtype != np.uint8:
                        if rgb.max() > 0:
                            rgb = np.clip((rgb / rgb.max()) * 255.0, 0, 255).astype(np.uint8)
                        else:
                            rgb = rgb.astype(np.uint8)

                    # Extract geographic bounds and resolution
                    bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                    res = (abs(src.transform.a), abs(src.transform.e))

                    # Transform bounds to lat/lon (WGS84) if projected
                    bounds_latlon = None
                    if is_geo and src.crs:
                        try:
                            wgs84 = CRS.from_epsg(4326)
                            if src.crs != wgs84:
                                b = transform_bounds(src.crs, wgs84, *bounds)
                                bounds_latlon = (b[0], b[1], b[2], b[3])
                            else:
                                bounds_latlon = bounds
                        except Exception:
                            bounds_latlon = None

                    meta = GeoMetadata(
                        is_georeferenced=is_geo,
                        crs=crs,
                        transform=tuple(src.transform)[:6] if src.transform else None,
                        bounds=bounds if is_geo else None,
                        bounds_latlon=bounds_latlon,
                        resolution=res if is_geo else None,
                        width=src.width,
                        height=src.height,
                        channels=3,
                        nodata=src.nodata,
                        driver=src.driver,
                    )
                    return rgb, meta
            except Exception:
                # If rasterio fails on non-geo image, fall through to Pillow
                pass

        # Standard image reading via Pillow
        pil_img = Image.open(file_path).convert("RGB")
        arr = np.asarray(pil_img, dtype=np.uint8)
        w, h = pil_img.size
        meta = GeoMetadata(is_georeferenced=False, width=w, height=h, channels=3)
        return arr, meta

    # Case 2: PIL Image instance
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        arr = np.asarray(pil_img, dtype=np.uint8)
        w, h = pil_img.size
        return arr, GeoMetadata(is_georeferenced=False, width=w, height=h, channels=3)

    # Case 3: NumPy array
    elif isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        h, w = arr.shape[:2]
        return arr.astype(np.uint8), GeoMetadata(is_georeferenced=False, width=w, height=h, channels=3)

    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def extract_tiles(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
) -> List[TileCrop]:
    """Splits an image into overlapping sliding-window tiles.

    Args:
        image: Array of shape (H, W, C) or (H, W).
        tile_size: Square window dimension in pixels (default: 512).
        overlap: Overlap border in pixels between adjacent tiles (default: 64).

    Returns:
        List[TileCrop]: List of tiles with image crops and source bounding boxes (y0, y1, x0, x1).
    """
    h, w = image.shape[:2]

    # If image fits within a single tile, return as single crop
    if h <= tile_size and w <= tile_size:
        return [TileCrop(crop=image, bbox=(0, h, 0, w), tile_index=0)]

    stride = max(1, tile_size - overlap)

    # Calculate tile top-left start positions
    y_starts = list(range(0, h - tile_size + 1, stride))
    if not y_starts or y_starts[-1] + tile_size < h:
        y_starts.append(max(0, h - tile_size))

    x_starts = list(range(0, w - tile_size + 1, stride))
    if not x_starts or x_starts[-1] + tile_size < w:
        x_starts.append(max(0, w - tile_size))

    # Remove duplicates preserving order
    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    tiles: List[TileCrop] = []
    idx = 0
    for y0 in y_starts:
        y1 = min(h, y0 + tile_size)
        for x0 in x_starts:
            x1 = min(w, x0 + tile_size)
            crop = image[y0:y1, x0:x1]
            tiles.append(TileCrop(crop=crop, bbox=(y0, y1, x0, x1), tile_index=idx))
            idx += 1

    return tiles


def generate_blend_mask(
    height: int,
    width: int,
    overlap: int,
    method: str = "linear",
) -> np.ndarray:
    """Generates a 2D feathering weight mask for seamless boundary blending.

    Args:
        height: Tile height in pixels.
        width: Tile width in pixels.
        overlap: Overlap margin width in pixels.
        method: "linear" or "cosine" feathering curve.

    Returns:
        np.ndarray: Float32 weight array of shape (height, width) with values in [0.0, 1.0].
    """
    if overlap <= 0:
        return np.ones((height, width), dtype=np.float32)

    def ramp(length: int, pad: int) -> np.ndarray:
        if length <= 2 * pad:
            pad = max(1, length // 2)
        r = np.ones(length, dtype=np.float32)
        idx = np.arange(pad, dtype=np.float32)
        if method == "cosine":
            factor = 0.5 * (1.0 - np.cos(np.pi * idx / pad))
        else:  # linear
            factor = (idx + 1.0) / (pad + 1.0)
        r[:pad] = factor
        r[-pad:] = factor[::-1]
        return r

    weight_y = ramp(height, overlap)[:, None]
    weight_x = ramp(width, overlap)[None, :]
    weight_2d = (weight_y * weight_x).astype(np.float32)
    # Ensure minimum non-zero weight to avoid division by zero
    return np.maximum(weight_2d, 1e-4)


def stitch_tiles(
    tile_depths: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
    output_shape: Tuple[int, int],
    overlap: int = 64,
    blend_method: str = "linear",
) -> np.ndarray:
    """Reassembles overlapping depth prediction tiles into a single continuous DSM.

    Applies weighted blend accumulation across tile borders to remove seams.

    Args:
        tile_depths: List of tuples (depth_array, (y0, y1, x0, x1)).
        output_shape: Target full image dimensions (H, W).
        overlap: Overlap in pixels.
        blend_method: "linear" or "cosine".

    Returns:
        np.ndarray: Reassembled float32 depth map of shape (H, W).
    """
    h, w = output_shape
    accum_depth = np.zeros((h, w), dtype=np.float32)
    accum_weight = np.zeros((h, w), dtype=np.float32)

    for depth_tile, (y0, y1, x0, x1) in tile_depths:
        th, tw = y1 - y0, x1 - x0
        # If tile depth size doesn't match bbox, resize it
        if depth_tile.shape != (th, tw):
            d_img = Image.fromarray(depth_tile.astype(np.float32)).resize(
                (tw, th), Image.Resampling.BICUBIC
            )
            d_tile = np.asarray(d_img, dtype=np.float32)
        else:
            d_tile = depth_tile.astype(np.float32)

        # Generate feathering weight mask for this tile
        weight_mask = generate_blend_mask(th, tw, overlap=overlap, method=blend_method)

        accum_depth[y0:y1, x0:x1] += d_tile * weight_mask
        accum_weight[y0:y1, x0:x1] += weight_mask

    # Normalize by accumulated weights
    valid_mask = accum_weight > 0
    accum_depth[valid_mask] /= accum_weight[valid_mask]

    return accum_depth


def main() -> None:
    """CLI runner for inspecting images and demonstrating sliding-window tiling."""
    parser = argparse.ArgumentParser(
        description="DepthWizard Task T3: Inspect image metadata and demonstrate sliding-window tiling."
    )
    parser.add_argument("image_path", type=str, help="Path to 2D image (PNG/JPG/GeoTIFF).")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size in pixels (default: 512).")
    parser.add_argument("--overlap", type=int, default=64, help="Tile overlap in pixels (default: 64).")
    args = parser.parse_args()

    input_path = Path(args.image_path)
    if not input_path.exists():
        print(f"Error: Image '{input_path}' not found.")
        return

    print("--- Inspecting Image ---")
    rgb, meta = inspect_image(input_path)
    print(meta.summary())

    print("\n--- Sliding-Window Tiling Breakdown ---")
    tiles = extract_tiles(rgb, tile_size=args.tile_size, overlap=args.overlap)
    print(f"Total tiles generated: {len(tiles)} (tile_size={args.tile_size}px, overlap={args.overlap}px)")
    for t in tiles[:5]:  # show first 5
        print(f"  Tile #{t.tile_index}: bbox (y0={t.bbox[0]}, y1={t.bbox[1]}, x0={t.bbox[2]}, x1={t.bbox[3]}), shape={t.crop.shape}")
    if len(tiles) > 5:
        print(f"  ... and {len(tiles) - 5} more tiles.")


if __name__ == "__main__":
    main()
