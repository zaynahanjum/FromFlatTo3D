"""FastAPI Backend & Pipeline Orchestration Module (Task Card T8).

Connects all processing stages into an end-to-end service:
1. Receives uploaded satellite/aerial image (PNG, JPG, GeoTIFF).
2. Performs geospatial inspection and sliding-window tiling (src.preprocess).
3. Executes monocular relative depth estimation (src.depth_model).
4. Fits metric scale and datum (src.calibrate) using GCP landmarks or preset bounds.
5. Exports multi-format DSMs (src.dsm_export) and updates viewer/public/ assets.
6. Returns JSON with 3D texture URLs, heightmap URLs, and metric elevation statistics.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.calibrate import (
    CalibrationResult,
    calibrate_with_gcps,
    calibrate_with_preset,
)
from src.depth_model import DepthEstimator, predict_relative_depth
from src.dsm_export import export_dsm
from src.preprocess import GeoMetadata, extract_tiles, inspect_image, stitch_tiles

# Try importing FastAPI
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def process_image_pipeline(
    image_path: Union[str, Path],
    base_elevation: Optional[float] = None,
    max_height_range: Optional[float] = None,
    gcps: Optional[List[Tuple[int, int, float]]] = None,
    output_dir: Union[str, Path] = "data/outputs",
    viewer_dir: Union[str, Path] = "viewer/public",
    tile_size: int = 512,
    overlap: int = 64,
) -> Dict[str, Any]:
    """Orchestrates the entire single-view height estimation pipeline.

    Args:
        image_path: Path to input 2D image.
        base_elevation: Optional base ground level in meters.
        max_height_range: Optional height span in meters.
        gcps: Optional list of (row, col, height_meters) landmarks.
        output_dir: Directory to save scientific data outputs (.npy, .tif, .json).
        viewer_dir: Directory where Three.js viewer loads public assets.
        tile_size: Tiling dimension for large satellite images.
        overlap: Tile overlap margin in pixels.

    Returns:
        Dict[str, Any]: Comprehensive pipeline results payload.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    stem = img_path.stem
    out_dir = Path(output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    v_dir = Path(viewer_dir)
    v_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Pipeline] 1. Inspecting image: {img_path.name}...")
    rgb_arr, geo_meta = inspect_image(img_path)
    h, w = rgb_arr.shape[:2]
    pil_img = Image.fromarray(rgb_arr)

    print(f"[Pipeline] 2. Estimating relative depth (shape: {w}x{h} px)...")
    if h > tile_size or w > tile_size:
        print(f"[Pipeline]    Large image detected: tiling into {tile_size}px overlapping windows...")
        tiles = extract_tiles(rgb_arr, tile_size=tile_size, overlap=overlap)
        estimator = DepthEstimator()
        tile_depths = []
        for t in tiles:
            d_tile, _ = estimator.predict_relative_depth(t.crop)
            tile_depths.append((d_tile, t.bbox))
        rdsm = stitch_tiles(tile_depths, (h, w), overlap=overlap)
    else:
        rdsm, _ = predict_relative_depth(pil_img)

    print(f"[Pipeline] 3. Calibrating metric scale...")
    if gcps and len(gcps) >= 2:
        calib = calibrate_with_gcps(rdsm, gcps)
    else:
        calib = calibrate_with_preset(
            rdsm,
            base_elevation_m=base_elevation,
            max_height_range_m=max_height_range,
        )
    print(f"[Pipeline]    Equation: Z = ({calib.a:.2f}) * d + ({calib.b:.2f}) m (mode={calib.mode})")

    print(f"[Pipeline] 4. Exporting DSMs...")
    export_prefix = out_dir / stem
    exported = export_dsm(
        calib.dsm,
        output_prefix=export_prefix,
        geo_meta=geo_meta,
        reference_image_path=img_path,
        metadata_extra={"calibration_mode": calib.mode, "equation": f"Z = {calib.a:.2f} * d + {calib.b:.2f}"},
    )

    print(f"[Pipeline] 5. Updating 3D viewer assets...")
    # Copy texture and displacement map to viewer/public/
    viewer_rgb = v_dir / f"{stem}_rgb.png"
    viewer_depth = v_dir / f"{stem}_depth.png"
    pil_img.save(viewer_rgb)
    if "depth_8bit" in exported:
        shutil.copyfile(exported["depth_8bit"], viewer_depth)

    payload = {
        "status": "success",
        "image_name": stem,
        "mode": "DSM" if calib.is_metric else "rDSM",
        "is_georeferenced": geo_meta.is_georeferenced,
        "crs": geo_meta.crs,
        "texture_url": f"public/{stem}_rgb.png",
        "displacement_url": f"public/{stem}_depth.png",
        "elevation_stats": {
            "base_elevation_m": round(float(calib.b), 2),
            "max_height_range_m": round(float(calib.a), 2),
            "min_elevation_m": round(float(calib.dsm.min()), 2),
            "max_elevation_m": round(float(calib.dsm.max()), 2),
            "mean_elevation_m": round(float(calib.dsm.mean()), 2),
            "vertical_span_m": round(float(calib.dsm.max() - calib.dsm.min()), 2),
        },
        "calibration": {
            "mode": calib.mode,
            "equation": f"Z = ({calib.a:.3f}) * depth + ({calib.b:.3f}) meters",
            "r2": round(float(calib.r2), 4),
            "fit_rmse_m": round(float(calib.fit_rmse), 2),
            "holdout_rmse_m": round(float(calib.holdout_rmse), 2) if calib.holdout_rmse else None,
        },
        "artifacts": {k: str(v) for k, v in exported.items()},
    }
    return payload


# Initialize FastAPI application if installed
if HAS_FASTAPI:
    app = FastAPI(
        title="Flat to 3D API",
        description="Single-View Satellite Height Estimation and 3D Flythrough (SIH 2026 - PS SIH26175)",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static assets from viewer directory
    viewer_path = Path("viewer")
    if viewer_path.exists():
        app.mount("/viewer", StaticFiles(directory="viewer", html=True), name="viewer")

    @app.get("/")
    def root() -> Dict[str, str]:
        return {
            "service": "Flat to 3D Single-View Height Estimation API",
            "status": "online",
            "version": "1.0.0",
            "docs": "/docs",
            "viewer": "/viewer",
        }

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "model": "Depth-Anything-V2-Small-hf",
            "tiling_ready": True,
            "calibration_ready": True,
        }

    @app.post("/process")
    async def process_endpoint(
        file: UploadFile = File(...),
        base_elevation: Optional[float] = Form(None),
        max_height_range: Optional[float] = Form(None),
        gcps_json: Optional[str] = Form(None),
    ) -> Dict[str, Any]:
        """Uploads an aerial/satellite image, generates metric DSM, and returns 3D viewer metadata."""
        upload_dir = Path("data/raw/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        filename = file.filename or "upload.png"
        saved_file_path = upload_dir / filename

        with open(saved_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        parsed_gcps = None
        if gcps_json:
            try:
                raw_gcps = json.loads(gcps_json)
                parsed_gcps = [(int(p[0]), int(p[1]), float(p[2])) for p in raw_gcps]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid gcps_json format: {e}")

        try:
            result = process_image_pipeline(
                image_path=saved_file_path,
                base_elevation=base_elevation,
                max_height_range=max_height_range,
                gcps=parsed_gcps,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {e}")

else:
    app = None


def main() -> None:
    """CLI runner for direct pipeline execution or server launch."""
    parser = argparse.ArgumentParser(
        description="Flat to 3D: Pipeline Orchestration & API Runner."
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run end-to-end pipeline test on an input image path.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch FastAPI uvicorn server on port 8000.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000).")
    args = parser.parse_args()

    if args.test:
        test_path = Path(args.test)
        print(f"--- Running End-to-End Pipeline Test on: {test_path} ---")
        res = process_image_pipeline(test_path)
        print("\n--- Pipeline Completed Successfully ---")
        print(json.dumps(res, indent=2))

    elif args.serve:
        if not HAS_FASTAPI:
            print("Error: FastAPI and Uvicorn must be installed to run the server.")
            print("Install them via: pip install fastapi uvicorn python-multipart")
            return
        import uvicorn

        print(f"Starting Flat to 3D API server at http://127.0.0.1:{args.port}...")
        uvicorn.run("src.api:app", host="127.0.0.1", port=args.port, reload=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
