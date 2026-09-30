# DepthWizard (SIH26175) - Day-0 MVP

Single-view satellite height estimation and interactive 3D WebGL flythrough.

---

## Quickstart (Day-0 MVP)

### 1. Requirements & Setup
- Python 3.10+ (or virtual environment)
- Dependencies: `torch`, `transformers`, `pillow`, `numpy`, `accelerate`

Ensure `.venv` is active:
```bash
source .venv/bin/activate
```

### 2. Prepare Sample Aerial Image
Generate a synthetic test aerial city tile:
```bash
python src/create_sample_tile.py
```
*(Or drop any `.png` / `.jpg` satellite image into `data/raw/`)*

### 3. Run Depth Inference
Run Depth Anything V2 Small on the satellite image:
```bash
python src/make_depth.py data/raw/sample_city.png
```
This generates:
- `viewer/public/sample_city_rgb.png` (satellite texture)
- `viewer/public/sample_city_depth.png` (normalized height displacement map)

### 4. Open 3D Viewer
Start any local static server (e.g., Python HTTP server):
```bash
python3 -m http.server 8000 --directory viewer
```
Then open `http://localhost:8000` in your web browser.

### 5. Features in the Viewer
- **3D Orbit & Pan**: Left-click + drag to orbit, right-click + drag to pan, scroll to zoom.
- **Click-to-Measure**: Click anywhere on the terrain surface to inspect exact total elevation, relative depth, and height above ground.
- **Displacement Slider**: Adjust 3D terrain exaggeration live.
- **Base Elevation & Max Height Sliders**: Calibrate ground elevation and structure height range.
- **Wireframe Toggle**: Inspect 3D polygon topology.
