"""Metric Scale Calibration & Ground Control Point (GCP) Fitting Module (Task Cards T5 & T14).

Converts relative depth d(x, y) in [0.0, 1.0] into real-world elevation in meters:
    Z(x, y) = a * d(x, y) + b

Supports:
1. Ground Control Point (GCP) Mode (Task T14):
   - Fits a (scale) and b (ground datum) from landmark points with known elevations.
   - Uses robust Huber regression to prevent outlier contamination.
   - Computes Hold-out RMSE on a 20% test split to satisfy the Hackathon's "Measurable Trust" criterion.
2. Coarse DEM Mode:
   - Fits against aligned Copernicus GLO-30 or SRTM elevation rasters.
3. Fallback Preset Mode:
   - Uses base elevation and max height range from config.yaml.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

# Try importing scikit-learn robust regression
try:
    from sklearn.linear_model import HuberRegressor

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


DEFAULT_CALIBRATION_CONFIG: Dict[str, Any] = {
    "default_base_elevation_m": 50.0,
    "default_max_height_range_m": 150.0,
    "default_displacement_scale": 35.0,
}


@dataclass
class CalibrationResult:
    """Encapsulates metric DSM calibration parameters and quality evaluation metrics."""

    a: float  # Scale factor (vertical height range in meters)
    b: float  # Datum offset (ground base elevation in meters)
    r2: float  # Coefficient of determination R^2
    fit_rmse: float  # RMSE on fitted calibration points (meters)
    holdout_rmse: Optional[float]  # RMSE on held-out check points (meters)
    dsm: np.ndarray  # Calibrated Digital Surface Model (H, W) in meters
    mode: str  # "gcp", "dem", or "preset"
    is_metric: bool  # True if calibrated with real reference measurements
    num_points_used: int = 0

    def summary(self) -> str:
        """Returns a formatted human-readable summary of calibration quality."""
        holdout_str = (
            f"{self.holdout_rmse:.2f} m"
            if self.holdout_rmse is not None
            else "N/A (insufficient points for split)"
        )
        return (
            f"--- Metric Calibration Summary ({self.mode.upper()} Mode) ---\n"
            f"  - Equation:        Z = ({self.a:.3f}) * depth + ({self.b:.3f}) meters\n"
            f"  - Base Elevation:  {self.b:.2f} m\n"
            f"  - Max Height Span: {self.a:.2f} m\n"
            f"  - Peak Elevation:  {self.dsm.max():.2f} m\n"
            f"  - Points Used:     {self.num_points_used}\n"
            f"  - Goodness of Fit: R² = {self.r2:.4f}\n"
            f"  - Fit RMSE:        {self.fit_rmse:.2f} m\n"
            f"  - Hold-out RMSE:   {holdout_str} (Measurable Trust Metric)"
        )


def load_calibration_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Loads calibration parameters from config.yaml with defaults."""
    cfg = DEFAULT_CALIBRATION_CONFIG.copy()
    path = Path(config_path)
    if path.exists() and yaml is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "calibration" in data:
                    cfg.update(data["calibration"])
        except Exception:
            pass
    return cfg


def robust_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    delta: float = 1.35,
    max_iter: int = 50,
) -> Tuple[float, float, float, float]:
    """Fits y = a * x + b using robust Huber regression (or IRLS fallback).

    Args:
        x: 1D array of predictor values (relative depth [0..1]).
        y: 1D array of target values (elevation in meters).
        delta: Huber threshold parameter.
        max_iter: Maximum IRLS iterations if scikit-learn is unavailable.

    Returns:
        Tuple[float, float, float, float]: (slope 'a', intercept 'b', R^2, RMSE).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(x)

    if n < 2:
        raise ValueError("At least 2 points are required for linear calibration.")

    if HAS_SKLEARN:
        reg = HuberRegressor(epsilon=delta, max_iter=200)
        reg.fit(x.reshape(-1, 1), y)
        a = float(reg.coef_[0])
        b = float(reg.intercept_)
    else:
        # Robust Iteratively Reweighted Least Squares (IRLS) using Huber weights
        # Initial estimate with Ordinary Least Squares
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        var_x = np.sum((x - x_mean) ** 2)
        if var_x < 1e-8:
            a = 1.0
            b = y_mean
        else:
            a = float(np.sum((x - x_mean) * (y - y_mean)) / var_x)
            b = float(y_mean - a * x_mean)

        weights = np.ones(n, dtype=np.float64)
        for _ in range(max_iter):
            residuals = y - (a * x + b)
            # Estimate scale with Median Absolute Deviation (MAD)
            mad = np.median(np.abs(residuals - np.median(residuals)))
            scale = 1.4826 * mad if mad > 1e-6 else 1.0

            u = np.abs(residuals) / (scale * delta + 1e-8)
            weights = np.where(u <= 1.0, 1.0, 1.0 / (u + 1e-8))

            # Weighted least squares update
            W = np.sum(weights)
            wx = np.sum(weights * x)
            wy = np.sum(weights * y)
            wxx = np.sum(weights * x * x)
            wxy = np.sum(weights * x * y)

            denom = W * wxx - wx * wx
            if abs(denom) < 1e-8:
                break
            a_new = (W * wxy - wx * wy) / denom
            b_new = (wy - a_new * wx) / W

            if abs(a_new - a) < 1e-5 and abs(b_new - b) < 1e-5:
                a, b = float(a_new), float(b_new)
                break
            a, b = float(a_new), float(b_new)

    # Calculate metrics
    y_pred = a * x + b
    res = y - y_pred
    rmse = float(np.sqrt(np.mean(res**2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(res**2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 1.0

    return a, b, r2, rmse


def calibrate_with_gcps(
    depth: np.ndarray,
    gcps: List[Tuple[int, int, float]],
    test_ratio: float = 0.2,
) -> CalibrationResult:
    """Calibrates relative depth to metric elevation using Ground Control Points (Task T14).

    Args:
        depth: Float32 array of shape (H, W) in [0.0, 1.0].
        gcps: List of tuples (row_idx, col_idx, real_height_meters).
        test_ratio: Fraction of points held out for independent RMSE validation (default: 0.2).

    Returns:
        CalibrationResult: Metric DSM and validation scores.
    """
    if len(gcps) < 2:
        raise ValueError("At least 2 Ground Control Points are required to fit scale and shift.")

    h, w = depth.shape[:2]
    sampled_depths = []
    real_heights = []

    for r, c, elev_m in gcps:
        r_clamp = min(max(0, int(r)), h - 1)
        c_clamp = min(max(0, int(c)), w - 1)
        sampled_depths.append(float(depth[r_clamp, c_clamp]))
        real_heights.append(float(elev_m))

    d_arr = np.array(sampled_depths, dtype=np.float64)
    z_arr = np.array(real_heights, dtype=np.float64)
    n = len(d_arr)

    # Hold-out validation split if sufficient points exist (>= 4 points)
    holdout_rmse = None
    if n >= 4 and test_ratio > 0:
        # Deterministic interleaved split
        test_mask = np.zeros(n, dtype=bool)
        test_indices = np.arange(1, n, int(1.0 / test_ratio))[: max(1, int(n * test_ratio))]
        test_mask[test_indices] = True
        train_mask = ~test_mask

        d_train, z_train = d_arr[train_mask], z_arr[train_mask]
        d_test, z_test = d_arr[test_mask], z_arr[test_mask]

        a, b, r2, fit_rmse = robust_linear_fit(d_train, z_train)

        # Evaluate on hold-out check set
        z_test_pred = a * d_test + b
        holdout_rmse = float(np.sqrt(np.mean((z_test - z_test_pred) ** 2)))
    else:
        a, b, r2, fit_rmse = robust_linear_fit(d_arr, z_arr)

    # Compute metric DSM across the entire image grid
    metric_dsm = (a * depth + b).astype(np.float32)

    return CalibrationResult(
        a=a,
        b=b,
        r2=r2,
        fit_rmse=fit_rmse,
        holdout_rmse=holdout_rmse,
        dsm=metric_dsm,
        mode="gcp",
        is_metric=True,
        num_points_used=n,
    )


def calibrate_with_dem(
    depth: np.ndarray,
    dem: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> CalibrationResult:
    """Calibrates relative depth against a resampled coarse reference DEM (Task T5).

    Args:
        depth: Relative depth array (H, W) in [0.0, 1.0].
        dem: Coarse DEM array resampled to (H, W) in meters.
        mask: Optional boolean mask (H, W), True where valid elevation exists.

    Returns:
        CalibrationResult: Metric DSM and fit statistics.
    """
    if depth.shape != dem.shape:
        raise ValueError(f"Shape mismatch: depth {depth.shape} vs dem {dem.shape}")

    d_flat = depth.ravel()
    e_flat = dem.ravel()

    if mask is not None:
        m_flat = mask.ravel()
        d_flat = d_flat[m_flat]
        e_flat = e_flat[m_flat]

    # Filter out NaNs or non-finite values
    valid = np.isfinite(d_flat) & np.isfinite(e_flat)
    d_valid = d_flat[valid]
    e_valid = e_flat[valid]

    # Subsample if dataset is very large (> 20,000 pixels) for fast robust fitting
    if len(d_valid) > 20000:
        step = len(d_valid) // 10000
        d_sub = d_valid[::step]
        e_sub = e_valid[::step]
    else:
        d_sub, e_sub = d_valid, e_valid

    a, b, r2, rmse = robust_linear_fit(d_sub, e_sub)
    metric_dsm = (a * depth + b).astype(np.float32)

    return CalibrationResult(
        a=a,
        b=b,
        r2=r2,
        fit_rmse=rmse,
        holdout_rmse=None,
        dsm=metric_dsm,
        mode="dem",
        is_metric=True,
        num_points_used=len(d_sub),
    )


def calibrate_with_preset(
    depth: np.ndarray,
    base_elevation_m: Optional[float] = None,
    max_height_range_m: Optional[float] = None,
    config_path: Union[str, Path] = "config.yaml",
) -> CalibrationResult:
    """Fallback calibration mapping [0.0, 1.0] to configured base and height range.

    Args:
        depth: Float32 array (H, W) in [0.0, 1.0].
        base_elevation_m: Base elevation in meters. If None, read from config.
        max_height_range_m: Vertical span in meters. If None, read from config.
        config_path: Path to config.yaml.

    Returns:
        CalibrationResult: Metric DSM with labeled preset parameters.
    """
    cfg = load_calibration_config(config_path)
    base = (
        base_elevation_m
        if base_elevation_m is not None
        else float(cfg.get("default_base_elevation_m", 50.0))
    )
    span = (
        max_height_range_m
        if max_height_range_m is not None
        else float(cfg.get("default_max_height_range_m", 150.0))
    )

    metric_dsm = (base + depth * span).astype(np.float32)

    return CalibrationResult(
        a=span,
        b=base,
        r2=1.0,
        fit_rmse=0.0,
        holdout_rmse=None,
        dsm=metric_dsm,
        mode="preset",
        is_metric=False,  # Uncalibrated baseline preset
        num_points_used=0,
    )


def main() -> None:
    """CLI runner for testing metric calibration with GCPs or presets."""
    parser = argparse.ArgumentParser(
        description="DepthWizard Task T5/T14: Calibrate relative depth to metric elevation in meters."
    )
    parser.add_argument("depth_npy", type=str, help="Path to input relative depth (.npy) file.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/outputs/calibrated_dsm.npy",
        help="Path to save calibrated metric DSM (.npy).",
    )
    parser.add_argument(
        "--base",
        type=float,
        default=None,
        help="Override base elevation in meters (preset mode).",
    )
    parser.add_argument(
        "--range",
        type=float,
        default=None,
        help="Override max height range in meters (preset mode).",
    )

    args = parser.parse_args()
    input_path = Path(args.depth_npy)
    if not input_path.exists():
        print(f"Error: Depth file '{input_path}' not found.")
        return

    depth = np.load(input_path).astype(np.float32)
    print(f"Loaded depth array: shape={depth.shape}, min={depth.min():.4f}, max={depth.max():.4f}")

    # Demonstrate GCP mode with sample landmarks (representing ground + buildings + skyscrapers)
    sample_gcps = [
        (384, 384, 185.0),  # Central Skyscraper tower
        (200, 200, 110.0),  # Mid-rise commercial block
        (550, 200, 85.0),   # Low-rise residential block
        (650, 650, 52.0),   # Ground / Street intersection
        (100, 650, 50.0),   # Ground / Perimeter road
    ]

    print("\n--- Running GCP Landmark Calibration (Task T14) ---")
    gcp_res = calibrate_with_gcps(depth, sample_gcps, test_ratio=0.2)
    print(gcp_res.summary())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, gcp_res.dsm)
    print(f"\nSaved calibrated metric DSM to: {out_path}")


if __name__ == "__main__":
    main()
