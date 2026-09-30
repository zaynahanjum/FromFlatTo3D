# Flat to 3D: Single-View Satellite Height Estimation & 3D Flythrough

> **Smart India Hackathon 2026** — Problem Statement **SIH26175**  
> **Theme:** Disaster Management | **Category:** Software  
> **Mission:** Convert ONE optical 2D aerial or satellite image (PNG/JPG/GeoTIFF) into a metric Digital Surface Model (DSM) and an interactive 3D WebGL flythrough with click-to-measure elevation and slope.

---

## Quick Start (Run in 1 Command)

### 1. Activate Environment & Install Dependencies
```bash
# Activate existing virtual environment
source .venv/bin/activate

# Install requirements (PyTorch, Transformers, FastAPI, Uvicorn, Pillow, NumPy)
pip install -r requirements.txt
```

### 2. Start the AI Server & 3D Viewer
```bash
uvicorn src.api:app --reload --port 8000
```

### 3. Open in Browser
Navigate to **[http://localhost:8000/](http://localhost:8000/)** (or `http://localhost:8000/viewer/app.html`) in your web browser.

---

## How It Works (The AI Pipeline)

Every image uploaded through the dashboard automatically passes through the end-to-end Python AI pipeline:

```
Input 2D Optical Image (PNG / JPG / GeoTIFF)
           │
           ▼
[1] src/preprocess.py   ──► Detects CRS, bounds, and splits large scenes into 512px overlapping tiles
           │
           ▼
[2] src/depth_model.py  ──► Vision Transformer (Depth-Anything-V2) predicts monocular relative depth
                            Automated water masking clamps water bodies to depressed basins
           │
           ▼
[3] src/calibrate.py    ──► Calibrates relative depth into real meters: Z = a * d + b
                            Fits Ground Control Points (GCPs) with robust Huber regression & hold-out RMSE
           │
           ▼
[4] src/dsm_export.py   ──► Exports:
                            • Single-band float32 GeoTIFF (.tif) for QGIS/GIS software
                            • 16-bit high-precision heightmap PNG for game engines (Unity / Unreal)
                            • 8-bit displacement map PNG for WebGL
                            • Metadata JSON with elevation statistics and spatial dimensions
           │
           ▼
[5] viewer/app.html     ──► Seamlessly reloads 3D terrain, scientific heatmap, and Terrain Inspector
```

---

## Interactive 3D Dashboard Features

- **Upload Image with Minimal Pill Overlay**:
  - Click **Upload Image** to select any satellite tile, aerial photo, or drone capture.
  - A modern, sleek centered loading pill displays processing status without visual clutter.
- **Click-to-Measure Elevation & Slope**:
  - Click anywhere on the terrain surface.
  - A purple measurement pin anchors to the clicked surface, and a floating callout displays:
    - **Total Elevation in meters** (e.g. `185.0 m elevation`)
    - **Feature Height above ground** (e.g. `+135.0 m above ground`)
    - **Local Surface Incline in degrees** (e.g. `Incline: 14.2°`) calculated via finite-difference gradient estimation:
      $$\text{slope} = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right) \times \frac{180^\circ}{\pi}$$
  - **Outside-Click Dismissal**: Clicking anywhere outside the terrain automatically dismisses the measurement marker and callout.
- **Terrain Inspector (Top-Left Dashboard)**:
  - **Core 2x2 Metrics Grid**: Base Ground, Peak Summit, Tallest Feature, and Average Height in meters.
  - **Elevation Profile Distribution**: Dynamic height distribution curve with minimum and maximum elevation axis markers.
  - **Universal Surface Relief Distribution**: Applicable to forests, cities, and natural landforms:
    - **Base Floor**: Lowlands, flat ground, roads, forest floor, or water bodies.
    - **Elevated Relief**: Tree canopy, buildings, cliffs, knolls, and ridges.
  - **Survey Dimensions & Scale**:
    - **Footprint Size**: Physical width and length in meters (e.g. `512 m × 512 m`).
    - **Covered Area**: Total surveyed ground area in square kilometers or hectares (e.g. `0.26 km²` / `26.2 ha`).
    - **Grid Resolution**: Image pixel dimensions and Ground Sampling Distance (e.g. `512 × 512 px (0.5 m/px)`).
    - **Vertical Relief**: Total height difference from lowest base to highest summit (e.g. `Δ 135.0 m`).
- **View Mode Switcher**:
  - **`Textured`**: Photorealistic optical aerial imagery.
  - **`Heatmap`**: 5-band Turbo-GIS scientific elevation colormap (Deep Navy $\to$ Bright Cyan $\to$ Vivid Green $\to$ Warm Amber $\to$ Fiery Crimson) for razor-sharp contour separation.
  - **`Wireframe`**: 3D geometric grid topology.
- **Camera Controls & Fullscreen**:
  - Orbit, zoom, and pan via mouse or touch controls.
  - Reset camera view (`↺`) and native browser Fullscreen toggle (`↗` / `↙`).
  - Height scale slider ($0 \dots 90$) for real-time terrain displacement adjustment.

---

## Repository Structure

```
depthSIH/
├── config.yaml                   # Central configuration (model, tiling, calibration, server)
├── requirements.txt              # Production dependencies
├── README.md                     # Master documentation
├── data/
│   ├── raw/                      # Input test imagery (sample_city.png, mountain_ridge.png, coastal_island.png)
│   └── outputs/                  # Exported DSMs (.tif, .npy, .png, .json)
├── src/
│   ├── __init__.py
│   ├── depth_model.py            # AI Monocular Depth (Depth-Anything-V2-Small-hf)
│   ├── preprocess.py             # Geospatial detection & sliding-window tiling
│   ├── calibrate.py              # Metric elevation scale fitting (GCP / DEM / Presets)
│   ├── dsm_export.py             # GeoTIFF and 16-bit DSM exporter
│   ├── api.py                    # FastAPI server & pipeline orchestrator
│   ├── create_sample_tile.py     # Generates synthetic 24-building cityscapes with shadows
│   └── create_landscape_samples.py # Generates mountain ridge and coastal island test tiles
└── viewer/
    ├── app.html                  # Full-screen Three.js WebGL 3D executive dashboard
    ├── index.html                # Standalone viewer fallback
    └── public/                   # Paired RGB textures and height displacement maps
```

---

## Standalone CLI Commands

You can run individual modules independently from the terminal:

### End-to-End Pipeline Test
```bash
python -m src.api --test data/raw/sample_city.png
```

### Standalone Depth Estimation
```bash
python -m src.depth_model data/raw/sample_city.png --output-dir data/outputs
```

### Standalone Preprocessing & Tiling
```bash
python -m src.preprocess data/raw/sample_city.png --tile-size 512 --overlap 64
```

### Standalone Metric Calibration (with GCPs)
```bash
python -m src.calibrate data/outputs/sample_city_rdsm.npy
```

### Standalone DSM Export
```bash
python -m src.dsm_export data/outputs/calibrated_dsm.npy --output-prefix data/outputs/my_dsm
```

---

## Testing with Sample Scenes

Pre-generated test scenes are available in `data/raw/`:
1. **`data/raw/sample_city.png`**: High-rise metropolis with stepped skyscrapers and cast shadows.
2. **`data/raw/mountain_ridge.png`**: Rugged mountainous terrain with steep slopes and valleys (ideal for testing slope angles).
3. **`data/raw/coastal_island.png`**: Ocean island with automatic water depression and plateau elevations.

---

## Governance & Git
- Git branch: `dev`
- Internal engineering plans, environment caches, and context documents are protected via `.gitignore`.
