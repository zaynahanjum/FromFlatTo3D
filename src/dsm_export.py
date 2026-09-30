"""Digital Surface Model (DSM) Export Module (Task Card T6).

Exports metric Digital Surface Models (DSMs) into standard geospatial and visualization formats:
1. Geospatial single-band float32 GeoTIFF (.tif) with embedded CRS, transform, and nodata value (-9999.0).
2. 16-bit high-precision heightmap PNG for game engine terrain importation (Unity / Unreal).
3. 8-bit WebGL displacement map PNG for Three.js rendering.
4. Raw float32 NumPy array (.npy) for scientific analysis.
5. Metric metadata JSON (.json) summarizing elevation statistics in meters.

Inputs:
    - dsm: 2D NumPy array of shape (H, W) in real-world meters.
    - geo_meta / reference_image: Optional spatial referencing (CRS, bounds, transform).
Outputs:
    - GeoTIFF (.tif), 16-bit PNG, 8-bit PNG, .npy, and .json files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from src.preprocess import GeoMetadata, inspect_image


def export_geotiff(
    dsm: np.ndarray,
    output_tif_path: Union[str, Path],
    crs: Any,
    transform: Any,
    nodata: float = -9999.0,
) -> Path:
    """Exports a 2D metric elevation array to a single-band float32 GeoTIFF.

    Args:
        dsm: Float32 array of shape (H, W) representing metric elevation in meters.
        output_tif_path: Path to target .tif file.
        crs: Coordinate Reference System (rasterio CRS object or EPSG string).
        transform: Affine transformation matrix.
        nodata: Nodata sentinel value (default: -9999.0).

    Returns:
        Path: Absolute path to written GeoTIFF file.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required to export georeferenced GeoTIFFs.")

    out_path = Path(output_tif_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = dsm.shape[:2]
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dsm.astype(np.float32), 1)

    return out_path


def export_dsm(
    dsm: np.ndarray,
    output_prefix: Union[str, Path],
    geo_meta: Optional[GeoMetadata] = None,
    reference_image_path: Optional[Union[str, Path]] = None,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Exports metric elevation models across standard GIS and visualization formats.

    Args:
        dsm: Float32 array of shape (H, W) in real-world meters.
        output_prefix: Target file path prefix without extension (parent dirs created automatically).
        geo_meta: Optional GeoMetadata instance describing geospatial CRS and transform.
        reference_image_path: Optional path to source satellite GeoTIFF to copy spatial metadata from.
        metadata_extra: Optional dictionary of additional properties to record in JSON.

    Returns:
        Dict[str, Path]: Map of output keys ('geotiff', 'depth_16bit', 'depth_8bit', 'npy', 'json') to paths.
    """
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    h, w = dsm.shape[:2]
    dsm_clean = np.where(np.isfinite(dsm), dsm, 0.0).astype(np.float32)

    min_elev = float(dsm_clean.min())
    max_elev = float(dsm_clean.max())
    mean_elev = float(dsm_clean.mean())
    std_elev = float(dsm_clean.std())
    span = max_elev - min_elev if (max_elev - min_elev) > 1e-6 else 1.0

    saved_files: Dict[str, Path] = {}

    # 1. Resolve geospatial metadata
    if geo_meta is None and reference_image_path is not None:
        try:
            _, geo_meta = inspect_image(reference_image_path)
        except Exception:
            geo_meta = None

    # 2. Export GeoTIFF if georeferenced and rasterio available
    if HAS_RASTERIO and geo_meta is not None and geo_meta.is_georeferenced and geo_meta.crs:
        try:
            tif_path = prefix.with_name(f"{prefix.name}_dsm.tif")
            crs_obj = CRS.from_string(geo_meta.crs) if isinstance(geo_meta.crs, str) else geo_meta.crs
            transform_obj = (
                Affine(*geo_meta.transform)
                if geo_meta.transform and not isinstance(geo_meta.transform, Affine)
                else geo_meta.transform
            )
            export_geotiff(dsm_clean, tif_path, crs=crs_obj, transform=transform_obj)
            saved_files["geotiff"] = tif_path
        except Exception as e:
            print(f"[DSM Export] Warning: Failed to write GeoTIFF: {e}")

    # 3. Export 16-bit Grayscale Heightmap PNG (normalized across vertical span)
    png_16bit_path = prefix.with_name(f"{prefix.name}_dsm_16bit.png")
    norm_16 = np.clip((dsm_clean - min_elev) / span * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(norm_16).save(png_16bit_path)
    saved_files["depth_16bit"] = png_16bit_path

    # 4. Export 8-bit Displacement Map PNG for WebGL
    png_8bit_path = prefix.with_name(f"{prefix.name}_dsm_8bit.png")
    norm_8 = np.clip((dsm_clean - min_elev) / span * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(norm_8, mode="L").save(png_8bit_path)
    saved_files["depth_8bit"] = png_8bit_path

    # 5. Export Raw Metric NumPy Array (.npy)
    npy_path = prefix.with_name(f"{prefix.name}_dsm.npy")
    np.save(npy_path, dsm_clean)
    saved_files["npy"] = npy_path

    # 6. Export Metadata JSON
    meta_json_path = prefix.with_name(f"{prefix.name}_meta.json")
    meta_dict: Dict[str, Any] = {
        "dimensions": {"width": w, "height": h},
        "elevation_units": "meters",
        "elevation_stats": {
            "min_m": round(min_elev, 2),
            "max_m": round(max_elev, 2),
            "mean_m": round(mean_elev, 2),
            "std_m": round(std_elev, 2),
            "vertical_span_m": round(span, 2),
        },
        "is_georeferenced": geo_meta.is_georeferenced if geo_meta else False,
        "crs": geo_meta.crs if geo_meta else None,
        "resolution": geo_meta.resolution if geo_meta else None,
        "bounds": geo_meta.bounds if geo_meta else None,
    }
    if metadata_extra:
        meta_dict.update(metadata_extra)

    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    saved_files["json"] = meta_json_path

    return saved_files


def main() -> None:
    """CLI runner for exporting metric DSM files."""
    parser = argparse.ArgumentParser(
        description="DepthWizard Task T6: Export calibrated metric DSM to GeoTIFF, 16-bit PNG, and JSON."
    )
    parser.add_argument("dsm_npy", type=str, help="Path to metric DSM array (.npy).")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="data/outputs/exported_dsm",
        help="Prefix path for exported artifacts.",
    )
    parser.add_argument(
        "--ref-image",
        type=str,
        default=None,
        help="Optional path to source GeoTIFF to copy spatial metadata from.",
    )
    args = parser.parse_args()

    input_path = Path(args.dsm_npy)
    if not input_path.exists():
        print(f"Error: DSM file '{input_path}' not found.")
        return

    dsm = np.load(input_path).astype(np.float32)
    print(f"Loaded DSM array: shape={dsm.shape}, min={dsm.min():.2f}m, max={dsm.max():.2f}m")

    artifacts = export_dsm(
        dsm,
        output_prefix=args.output_prefix,
        reference_image_path=args.ref_image,
    )

    print("\n--- DSM Export Completed ---")
    print("Generated Artifacts:")
    for k, p in artifacts.items():
        print(f"  [{k}]: {p}")


if __name__ == "__main__":
    main()
