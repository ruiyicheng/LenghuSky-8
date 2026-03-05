"""
Astrometric tiling solver for wide-field (fisheye/all-sky) frames
================================================================

Overview
--------
This module detects bright stars in a single grayscale frame, partitions the
field into HEALPix tiles, and performs a **local astrometric solve** per tile.
For each tile and projection type, it stores a minimal WCS header along with an
empirical error estimate. The resulting calibration is written to JSON and can
later be queried to convert image (x, y) pixel coordinates into ICRS
Right Ascension / Declination with a per-point uncertainty.

High-level pipeline
-------------------
1) **Load & pre-process image**
   - Read a color PNG/JPG, convert to grayscale, apply a gentle Gaussian blur to
     stabilize SEP background estimation and source extraction.

2) **Detection (SEP)**
   - Build a background model and extract sources with a configurable
     S/N threshold (`N_SIGMA_DETECTION`) and minimum area (`N_PIXEL_AREA`).
   - Optionally keep detections within a circular region centered on the
     presumed camera center (`CCAM`, `R`, `proportion`).

3) **Camera polar transform**
   - Convert detections (u, v) → polar (r, φ) around the camera center.
   - Convert r ↔ θ using a chosen fisheye projection model. Supported:
     `TAN`, `STG`, `ARC`, `ZEA`, `SIN`. The default list is in
     `PROJECTION_TYPE_LIST` (here only `SIN` is enabled).

4) **HEALPix grouping**
   - Map each detection’s (θ, φ) to a HEALPix index (`NSIDE` controls tile size).
   - Group detections by tile.

5) **Local astrometric solve (per tile)**
   - Rotate detections into the tile’s local frame (so they look tangent-plane-like).
   - Attempt an astrometric solve with a local index (astrometry.net series_4100),
     optionally given a **position hint** near the local zenith at observation time.
   - Solutions are stored as WCS headers if successful.

6) **Quality & error estimation**
   - Project the external **Gaia DR3 bright catalog** (CSV with `ra, dec`) through
     the solved WCS to pixel space, then nearest-neighbor match to detections
     with a KD-tree.
   - If enough matches are found and residuals are small, record a per-tile
     median residual (converted from pixels to degrees using local pixel scale).

7) **Persistence**
   - Write an easy-to-parse JSON blob with global metadata and per-tile/per-projection
     payloads (including WCS header cards). Writing is atomic.

8) **Querying**
   - `make_ra_dec_query_from_json(json_path)` returns a vectorized function
     `query(x_vec, y_vec) → (ra_deg, dec_deg, err_deg)`. It reproduces the same
     geometry used during solving: (u, v) → (r, φ) → (θ, φ) → tile selection →
     rotation into the tile’s local frame → (θ′, φ′) → (u′, v′) → WCS → (RA, Dec).
   - The first projection/tile that yields a valid result is used.

Coordinate systems & conventions
--------------------------------
- Image coordinates: `(u, v)` in **pixels**, origin at the top-left as read by OpenCV.
- Camera polar: `r = sqrt((u-u0)^2 + (v-v0)^2)`, `φ = atan2(v-v0, u-u0) ∈ [0, 2π)`.
- Fisheye projection models:
  - `TAN`:    θ = arctan(r/f)            (gnomonic)
  - `STG`:    θ = 2 arctan(r/(2f))       (stereographic)
  - `ARC`:    θ = r/f                     (equidistant)
  - `ZEA`:    θ = 2 asin(r/(2f))         (zenithal equal-area)
  - `SIN`:    θ = asin(r/f)               (orthographic)
- Spherical: HEALPix uses `(θ, φ)` with θ = colatitude ∈ [0, π], φ = longitude ∈ [0, 2π).
- Sky frame: WCS is in ICRS (RA, Dec) degrees.

Inputs
------
- **Images**: PNG/JPG paths. Filename must encode local UTC+8 time as
  `YYYY-MM-DD-hh-mm-ss` (used to compute zenith RA/Dec for the position hint).
- **Reference catalog**: CSV with columns `ra, dec` in degrees
  (e.g., Gaia DR3 bright stars; recommended magnitude cut ≲ 5.5).
- **Site**: latitude [deg], longitude [deg East], altitude [m].
- **Initial camera params**: approximate image center `INIT_CENTER` (u0, v0) and
  a scale `INIT_RADIUS` used as `f` in r↔θ projection formulae.

Outputs
-------
- **JSON** file per input image containing:
  - Global metadata: site, time (local and UTC ISO), image size, center, R,
    detection/tiling params, chosen projections, match threshold, etc.
  - For each **HEALPix tile** and **projection type**:
    - Solve inputs: tile center `(theta_center, phi_center)`, the detections’
      `(theta, phi)` in the *global* spherical frame.
    - Solve results (when successful):
      - `WCS`: minimal FITS-compatible header map (keys like CRVAL*, CRPIX*, CD*_*).
      - `err`: median Gaia↔detection residual in **degrees**.
      - `n_matches`: number of accepted matches used for the residual.

JSON schema (informal)
----------------------
{
  "meta": {
    "CENTER": [u0, v0],
    "R": float,
    "N_SIGMA_DETECTION": float,
    "N_PIXEL_AREA": int,
    "VALID_STAR_POS_PROPORTION": float,
    "NSIDE": int,
    "IMAGE_PATH": str,
    "IMAGE_SIZE": {"width": int, "height": int},
    "TIME_LOCAL_UTC_PLUS_8": "YYYY-MM-DD-hh-mm-ss",
    "TIME_UTC_ISO": "YYYY-MM-DDThh:mm:ss.sss",
    "SITE": {"lat_deg": float, "lon_deg": float, "alt_m": float},
    "PROJECTION_TYPES": [str, ...],
    "MATCH_THRESH_PX": float,
    "NOTE_ERR_DEG": str
  },
  "result": {
    "<healpix_id>": {
      "<proj_type>": {
        "theta": [float, ...],               # detections in global spherical frame
        "phi":   [float, ...],
        "theta_center": float,               # tile center (θ_c, φ_c)
        "phi_center": float,
        "n_stars": int,
        "err": float or NaN,                 # median residual in degrees
        "n_matches": int,
        "WCS": { "CRVAL1": ..., "CRVAL2": ..., "CRPIX1": ..., ... }  # when solved
      },
      ...
    },
    ...
  }
}

Key parameters & tuning
-----------------------
- `N_SIGMA_DETECTION`, `N_PIXEL_AREA`: adjust for your camera noise and star sizes.
- `INIT_CENTER`, `INIT_RADIUS`: coarse geometric guess; affects r↔θ mapping and
  therefore the HEALPix binning and rotation geometry. Reasonable values make
  solves easier but are not required to be perfect.
- `NSIDE`: larger values → smaller tiles; aim for ≥ 6 detections per tile.
- `MATCH_THRESH_PX`: KD-tree nearest-neighbor radius for Gaia↔detection matching.
  If set too small, you may undercount matches; too large may inflate residuals.
- `PROJECTION_TYPE_LIST`: use one or more models; the first successful result is
  used at query time.

Astrometry backend
------------------
Solving is delegated to an `astrometry` wrapper (astrometry.net indices 4100)
and cached in `ASTROMETRY_DIR`. The code passes up to the brightest ~50 local
detections per tile and may include a **position hint** derived from the site
and observation time (tile radius ≈ 110 deg around zenith).

Error model
-----------
Per-tile error is the **median** matched pixel residual converted to degrees
using `proj_plane_pixel_scales(WCS)`. Tiles with too few matches (≤ 8) or large
residuals (> 0.3 deg) are flagged as unsolved.

Query API
---------
- `make_ra_dec_query_from_json(json_path)`:
  - Returns a callable `query(x_vec, y_vec)` that:
    1) Transforms (x, y) → (r, φ) → (θ, φ) per projection.
    2) Picks the corresponding HEALPix tile.
    3) Rotates into the tile’s local frame (exactly as during solving).
    4) Maps (θ′, φ′) → (u′, v′) and applies the tile WCS.
    5) Returns `(ra_deg, dec_deg, err_deg)`, where `err_deg` is the stored tile error.
  - Vectorized: `x_vec` and `y_vec` must share the same shape.

File layout & defaults
----------------------
- Temporary/derived outputs: `TEMP_DIR` (created if missing).
- Astrometry cache: `ASTROMETRY_DIR` (created if missing).
- Inputs:
  - `INPUT_IMG_LIST`: glob over calibration images (PNG).
  - `REF_CAT_CSV`: path to Gaia DR3 bright subset CSV (`ra, dec` columns).

CLI usage (examples)
--------------------
- Batch process all images in `INPUT_IMG_LIST` and write one JSON per image:
  Run the module directly: `python this_script.py`.
- After writing JSON, you can reconstruct a query function and evaluate
  RA/Dec for arbitrary pixel grids (see the commented example at the bottom).

Dependencies
------------
- Python 3.9+ recommended
- numpy, pandas, scipy, OpenCV (`cv2`), SEP (`sep`), healpy, astropy,
  matplotlib, astrometry.net python wrapper

Troubleshooting
---------------
- **No stars detected**: relax `N_SIGMA_DETECTION`, reduce `N_PIXEL_AREA`,
  verify the image’s dynamic range and blur kernel.
- **Solve failed for many tiles**: ensure Gaia CSV exists and is bright-enough
  subset; increase `NSIDE` or `MATCH_THRESH_PX`; add more projection models;
  verify site/time parsing from filename matches local UTC+8 convention.
- **Large errors**: check `INIT_CENTER` and `INIT_RADIUS` sanity; bad WCS tiles
  are marked and won’t be used—confirm there are enough good tiles covering the
  region of interest.

License & attribution
---------------------
- Gaia DR3 data courtesy of ESA/Gaia/DPAC.
- Astrometric solving via astrometry.net indices (series_4100).
"""
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple, Callable

import numpy as np
import cv2
import sep
import healpy as hp
import pandas as pd
from scipy.spatial import cKDTree

import astropy
import astropy.wcs
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz
import astropy.units as units
from astropy.wcs.utils import proj_plane_pixel_scales
import matplotlib.pyplot as plt
import glob
# -----------------------------
# User / environment constants
# -----------------------------
UTC_PLUS_8 = timezone(timedelta(hours=8))
ASTROMETRY_DIR = "/mnt/hgfs/cloud/calibration/temp/astrometry_data"
os.makedirs(ASTROMETRY_DIR, exist_ok=True)

# INPUT_IMG_LIST   = ["/mnt/hgfs/cloud/data/calibration/2020/2020-01-01-00-02-40.png","/mnt/hgfs/cloud/data/calibration/2020/2020-01-01-00-08-17.png","/mnt/hgfs/cloud/data/calibration/2020/2020-01-01-01-15-38.png","/mnt/hgfs/cloud/data/calibration/2020/2020-01-01-01-21-15.png","/mnt/hgfs/cloud/data/calibration/2020/2020-01-01-01-26-52.png","/mnt/hgfs/cloud/data/calibration/2020/2020-01-01-01-32-28.png","/mnt/hgfs/cloud/data/calibration/2020/2020-01-01-01-38-05.png","/mnt/hgfs/cloud/data/calibration/2020/2020-01-01-02-39-49.png","/mnt/hgfs/cloud/data/calibration/2020/2020-01-01-03-07-53.png","/mnt/hgfs/cloud/data/calibration/2020/2020-01-01-03-47-10.png","/mnt/hgfs/cloud/data/calibration/2020/2020-01-01-04-54-31.png"]   # yyyy-mm-dd-hh-mm-ss.png (UTC+8)
INPUT_IMG_LIST = glob.glob("/mnt/hgfs/cloud/data/calibration/all_input/*.png")# [
#     "/mnt/hgfs/cloud/data/calibration/2018/2023-12-24-20-10-00.png",
# ]
REF_CAT_CSV = "/mnt/hgfs/cloud/data/input_cat/gaiadr3_bright.csv"               # must have 'ra','dec' (deg) Can be obtained from esa NASA archive, query the Gaiadr3 stars with mag<5.5

SITE_LAT = 38.9586     # deg (+N)
SITE_LON = 93.2681     # deg (+E)
SITE_ALT = 4200        # m
NSIDE = 4              # healpix nside for catalog partitioning
TEMP_DIR = "/mnt/hgfs/cloud/data/calibration/out"
os.makedirs(TEMP_DIR, exist_ok=True)

# Initial guess of FoV center & radius (pixels)
INIT_CENTER = (253*8, 251*8)#(251*8, 253*8)   # (u0, v0) in pixels
INIT_RADIUS = 233.0*8#213.0*8 #2098         # focal-like scale in px for the projection formulas

N_SIGMA_DETECTION = 10
N_PIXEL_AREA = 20
VALID_STAR_POS_PROPORTION = 0.95
PROJECTION_TYPE_LIST = ["SIN"]# ['TAN', 'STG', 'ARC', 'ZEA', 'SIN']

# Matching threshold (pixels) for Gaia↔detections KD-tree
MATCH_THRESH_PX = 12.0

# -----------------------------
# Utility transforms
# -----------------------------
def uv2rphi(u: np.ndarray, v: np.ndarray, u0: float, v0: float) -> Tuple[np.ndarray, np.ndarray]:
    du = u - u0
    dv = v - v0
    r = np.sqrt(du**2 + dv**2)
    phi = np.arctan2(dv, du)
    phi = (phi + 2*np.pi) % (2*np.pi)  # map to [0, 2π)
    return r, phi

def rphi2uv(r: np.ndarray, phi: np.ndarray, u0: float, v0: float) -> Tuple[np.ndarray, np.ndarray]:
    u = r * np.cos(phi) + u0
    v = r * np.sin(phi) + v0
    return u, v

def r2theta(r: np.ndarray, f: float, proj_type: str = 'TAN') -> np.ndarray:
    if proj_type == 'TAN':
        return np.arctan(r/f)
    elif proj_type == 'STG':
        return 2 * np.arctan(r/(2*f))
    elif proj_type == 'ARC':
        return r/f
    elif proj_type == 'ZEA':
        return 2 * np.arcsin(r/(2*f))
    elif proj_type == 'SIN':
        return np.arcsin(r/f)
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")

def theta2r(theta: np.ndarray, f: float, proj_type: str = 'TAN') -> np.ndarray:
    if proj_type == 'TAN':
        return f * np.tan(theta)
    elif proj_type == 'STG':
        return 2 * f * np.tan(theta/2)
    elif proj_type == 'ARC':
        return f * theta
    elif proj_type == 'ZEA':
        return 2 * f * np.sin(theta/2)
    elif proj_type == 'SIN':
        return f * np.sin(theta)
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")

# -----------------------------
# I/O helpers
# -----------------------------
def load_image(image_path: str) -> np.ndarray:
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, ksize=(15, 15), sigmaX=0)
    return gray

def to_astropy_time_utc8(ts: str, *, scale: str = "utc") -> Time:
    dt_local = datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=UTC_PLUS_8)
    return Time(dt_local.astimezone(timezone.utc), scale=scale)

def observation_conditions(time_str: str, SITE_LAT: float, SITE_LON: float, SITE_ALT: float):
    obstime = to_astropy_time_utc8(time_str)
    location = EarthLocation(lat=SITE_LAT*units.deg, lon=SITE_LON*units.deg, height=SITE_ALT*units.m)
    altaz_frame = AltAz(obstime=obstime, location=location)
    return obstime, location, altaz_frame

# -----------------------------
# Detection
# -----------------------------
def star_detection(image_gray: np.ndarray,
                   N_SIGMA_DETECTION: float = 5.0,
                   N_PIXEL_AREA: int = 5,
                   proportion: float = 0.95,
                   CCAM: Tuple[float, float] = None,
                   R: float = None) -> np.ndarray:
    try:
        img_data = image_gray.copy().astype(np.float32)
        bkg = sep.Background(image_gray)
    except Exception:
        img_data = image_gray.copy().byteswap().newbyteorder().astype(np.float32)
        bkg = sep.Background(image_gray)

    data_sub = img_data - bkg
    objects = sep.extract(data_sub, N_SIGMA_DETECTION, err=bkg.rms(), minarea=N_PIXEL_AREA)

    if CCAM is not None and R is not None:
        mask = np.sqrt((objects['x'] - CCAM[0])**2 + (objects['y'] - CCAM[1])**2) < proportion * R
        objects = objects[mask]

    objects.sort(order='flux')  # faint→bright, we’ll use the brightest via slice in solver
    # visualizaton
    # plt.figure(figsize=(10,10))
    # m = np.mean(data_sub)
    # s = np.std(data_sub)
    # plt.imshow(data_sub, cmap='gray', origin='lower', vmin=m - 1*s, vmax=m + 2*s)
    # plt.scatter(objects['x'], objects['y'], s=30, edgecolor='red', facecolor='none', lw=1.5)
    # plt.show()
    return objects

# -----------------------------
# Astrometric solve
# -----------------------------
def solve_astrometry_from_stars(pos_stars: np.ndarray,
                                cache_dir: str = ASTROMETRY_DIR,
                                position = None,
                                sip_order: int = 0) -> astropy.wcs.WCS:
    import astrometry  # local import to avoid issues if not installed elsewhere

    solver = astrometry.Solver(astrometry.series_4100.index_files(cache_directory=cache_dir, scales=[17,18,19]))
    params = astrometry.SolutionParameters(sip_order=sip_order, tune_up_logodds_threshold=None)
    position_hint = astrometry.PositionHint(ra_deg = position['ra'],
                                            dec_deg = position['dec'],
                                            radius_deg = position['radius']) if position is not None else None
    stars_feed = pos_stars[-50:] if pos_stars.shape[0] > 50 else pos_stars
    solution = solver.solve(stars=stars_feed,
                            size_hint=None,
                            position_hint=position_hint,
                            solution_parameters=params)
    if solution is None or not solution.has_match():
        raise RuntimeError("Astrometry solve failed.")
    return astropy.wcs.WCS(solution.best_match().wcs_fields)

def wcs_to_header_dict(w: astropy.wcs.WCS) -> Dict[str, Any]:
    hdr = w.to_header(relax=True)
    out = {}
    for k in hdr:
        v = hdr[k]
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = v
    return out

def header_dict_to_wcs(hdr_dict: Dict[str, Any]) -> astropy.wcs.WCS:
    hdr = fits.Header()
    for k, v in hdr_dict.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        hdr[k] = v
    return astropy.wcs.WCS(hdr)

# -----------------------------
# Neighbor fill helpers
# -----------------------------
_NUMERIC_WCS_KEYS = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"]

def _average_neighbor_wcs(headers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Average a list of WCS header dicts (>=2 recommended).
    Numeric keys in _NUMERIC_WCS_KEYS are averaged; other cards are copied from the first.
    """
    if not headers:
        return {}
    base = dict(headers[0])  # copy
    for key in _NUMERIC_WCS_KEYS:
        vals = []
        for h in headers:
            if key in h and isinstance(h[key], (int, float, np.integer, np.floating)):
                vals.append(float(h[key]))
        if len(vals) >= 1:
            base[key] = float(np.mean(vals))
    return base

# -----------------------------
# KD-tree error estimation
# -----------------------------
def _estimate_tile_error_deg(
    wcs: astropy.wcs.WCS,
    det_uv: np.ndarray,           # shape (N, 2), rotated detections for this tile
    gaia_ra: np.ndarray,          # shape (M,)
    gaia_dec: np.ndarray,         # shape (M,)
    img_w: int,
    img_h: int,
    match_thresh_px: float = MATCH_THRESH_PX
) -> Tuple[float, int]:
    """
    Project Gaia to pixel via WCS, match to detections in pixel space with KD-tree,
    accept matches within match_thresh_px. Return (median_residual_deg, n_matches).
    """
    if det_uv.size == 0 or gaia_ra.size == 0:
        return float("nan"), 0

    # Project Gaia to pixels
    world = np.column_stack([gaia_ra, gaia_dec])
    pix = wcs.wcs_world2pix(world, 0)  # [[x,y], ...]
    xg = pix[:, 0]
    yg = pix[:, 1]

    # Keep Gaia sources that land inside the image frame (robust bounds)
    inside = (xg >= np.min(det_uv[:,0])-50) & (xg < np.max(det_uv[:,0])+50) & (yg >= np.min(det_uv[:,1])-50) & (yg < np.max(det_uv[:,1])+50)
    xg = xg[inside]
    yg = yg[inside]
    if xg.size == 0:
        return float("nan"), 0
    # Visualize projected Gaia points and detections
    # plt.figure(figsize=(8,8))
    # plt.scatter(xg, yg, s=5, c='red', label='Projected Gaia', alpha=0.6)
    # plt.scatter(det_uv[:,0], det_uv[:,1], s=5, c='blue', label='Detections', alpha=0.6)
    # plt.xlim([np.min(det_uv[:,0])-50, np.max(det_uv[:,0])+50])
    # plt.ylim([np.min(det_uv[:,1])-50, np.max(det_uv[:,1])+50])
    # plt.legend()
    # plt.show()
    # KD-tree on detections
    tree = cKDTree(det_uv)  # det_uv is (N,2) of rotated_u,rotated_v for this tile
    dists, idx = tree.query(np.column_stack([xg, yg]), k=1, distance_upper_bound=match_thresh_px)

    # Valid matches are where distance is finite and < thresh
    valid = np.isfinite(dists) & (dists < match_thresh_px)
    if not np.any(valid):
        return float("nan"), 0
    print(f"Found {np.sum(valid)} matches out of {xg.size} projected Gaia sources.")
    # Convert pixel residuals to degrees using local pixel scale
    # proj_plane_pixel_scales returns degrees per pixel for each axis
    scales_deg_per_pix = proj_plane_pixel_scales(wcs)  # (deg/pix_x, deg/pix_y)
    mean_scale = float(np.mean(scales_deg_per_pix))
    resid_px = dists[valid]
    # Robust summarize → median residual (in degrees)
    resid_deg = float(np.median(resid_px) * mean_scale)
    return resid_deg, int(np.sum(valid))

# -----------------------------
# Main pipeline + JSON writer
# -----------------------------
def pipeline(image_path: str,
             input_catalog: str,
             SITE_LAT: float,
             SITE_LON: float,
             SITE_ALT: float,
             CCAM: Tuple[float, float],
             R: float,
             N_SIGMA_DETECTION: float = 5.0,
             N_PIXEL_AREA: int = 5,
             proportion: float = 0.95,
             nside: int = 4,
             json_out_path: Optional[str] = None) -> str:
    """
    Runs:
      - image load & detection
      - projection to (theta, phi) for each projection type
      - healpix grouping
      - per-group WCS solve on rotated set
      - error estimation via KD-tree Gaia↔detections (pixel threshold)
      - neighbor fill for missing tiles (WCS average)
      - write JSON with metadata & per-(healpix,proj) solution

    Returns path to JSON file.
    """
    # Output path
    if json_out_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        json_out_path = os.path.join(TEMP_DIR, f"{base}_astrometry.json")

    # Load image and size
    img_gray = load_image(image_path)
    img_h, img_w = img_gray.shape

    # Time & site (from filename yyyy-mm-dd-hh-mm-ss.png)
    timestamp_str = os.path.basename(image_path).split('.')[0]
    obstime, location, altaz_frame = observation_conditions(
        timestamp_str, SITE_LAT, SITE_LON, SITE_ALT)
    # Get zenith ra dec (alt = 90deg at this time & site)
    zenith = astropy.coordinates.SkyCoord(alt=90*units.deg, az=0*units.deg, frame=altaz_frame)
    print(f"Observation time (UTC): {obstime.utc.isot}, location: {location.to_geodetic()} zenith (RA,Dec): ({zenith.icrs.ra.deg:.6f}, {zenith.icrs.dec.deg:.6f}) deg")
    position_init = {'ra': float(zenith.icrs.ra.deg), 'dec': float(zenith.icrs.dec.deg), 'radius': 110.0}
    # --- Read Gaia bright catalog once
    Gaia_bright = pd.read_csv(input_catalog)  # must contain 'ra','dec' in degrees
    gaia_ra_all = Gaia_bright['ra'].to_numpy(dtype=float)
    gaia_dec_all = Gaia_bright['dec'].to_numpy(dtype=float)

    # Detect stars
    objects = star_detection(img_gray,
                             N_SIGMA_DETECTION=N_SIGMA_DETECTION,
                             N_PIXEL_AREA=N_PIXEL_AREA,
                             proportion=proportion,
                             CCAM=CCAM,
                             R=R)
    if len(objects) == 0:
        raise RuntimeError("No stars detected inside the search region.")

    # Pixel coords of detections
    u = objects['x']
    v = objects['y']
    r, phi = uv2rphi(u, v, CCAM[0], CCAM[1])

    # Shared metadata
    meta: Dict[str, Any] = {
        "CENTER": [float(CCAM[0]), float(CCAM[1])],
        "R": float(R),
        "N_SIGMA_DETECTION": float(N_SIGMA_DETECTION),
        "N_PIXEL_AREA": int(N_PIXEL_AREA),
        "VALID_STAR_POS_PROPORTION": float(proportion),
        "NSIDE": int(nside),
        "IMAGE_PATH": image_path,
        "IMAGE_SIZE": {"width": int(img_w), "height": int(img_h)},
        "TIME_LOCAL_UTC_PLUS_8": timestamp_str,
        "TIME_UTC_ISO": obstime.utc.isot,
        "SITE": {"lat_deg": float(SITE_LAT), "lon_deg": float(SITE_LON), "alt_m": float(SITE_ALT)},
        "PROJECTION_TYPES": list(PROJECTION_TYPE_LIST),
        "MATCH_THRESH_PX": float(MATCH_THRESH_PX),
        "NOTE_ERR_DEG": "Uncertainty estimated as median Gaia↔detection residual (deg) after projecting Gaia with the tile WCS."
    }

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    solved_mask: Dict[Tuple[int, str], bool] = {}

    # First pass: solve tiles and compute per-tile errors
    for proj_type in PROJECTION_TYPE_LIST:
        f0 = R
        theta = r2theta(r, f0, proj_type)  # theta in radians
        pix = hp.ang2pix(nside, theta, phi)

        # Group detections by pixel id
        groups: Dict[int, List[int]] = {}
        for i, p in enumerate(pix):
            p_int = int(p)
            groups.setdefault(p_int, []).append(i)

        for k, idxs in groups.items():
            print(f"Solving for proj {proj_type}, hpix {k} with {len(idxs)} stars")
            # Require minimum detections for a stable solve
            if len(idxs) < 6:
                solved_mask[(k, proj_type)] = False
                continue

            theta_this = theta[idxs]
            phi_this = phi[idxs]

            # Healpix center
            theta_c, phi_c = hp.pix2ang(nside, k)

            # Rotate to local frame
            xyz_this = hp.ang2vec(theta_this, phi_this)
            Rz = np.array([[np.cos(-phi_c), -np.sin(-phi_c), 0],
                           [np.sin(-phi_c),  np.cos(-phi_c), 0],
                           [0,                0,              1]])
            Ry = np.array([[np.cos(-theta_c), 0, np.sin(-theta_c)],
                           [0,                 1, 0],
                           [-np.sin(-theta_c), 0, np.cos(-theta_c)]])
            R_mat = Ry @ Rz
            xyz_rot = xyz_this @ R_mat.T
            rotated_theta, rotated_phi = hp.vec2ang(xyz_rot)

            # Back to pixel plane in original image coordinates
            rotated_r = theta2r(rotated_theta, f0, proj_type)
            rotated_u, rotated_v = rphi2uv(rotated_r, rotated_phi, CCAM[0], CCAM[1])

            # Try astrometric solve on local set
            try:
                pos_feed = np.column_stack([rotated_u, rotated_v])
                wcs = solve_astrometry_from_stars(pos_feed, cache_dir=ASTROMETRY_DIR, sip_order=4, position=position_init)
                solved = True
                err_deg, n_match = _estimate_tile_error_deg(
                    wcs=wcs,
                    det_uv=pos_feed,  # rotated detections for this tile
                    gaia_ra=gaia_ra_all,
                    gaia_dec=gaia_dec_all,
                    img_w=img_w,
                    img_h=img_h,
                    match_thresh_px=MATCH_THRESH_PX
                )

                print(err_deg, n_match)
                if err_deg > 0.3 or n_match <= 8:
                    print(f"  Warning: only {n_match} matches found for error estimation.")
                    solved = False
            except Exception:
                solved = False
                print(f"  Solve failed for proj {proj_type}, hpix {k}")

            node = results.setdefault(str(k), {})
            payload: Dict[str, Any] = {
                "theta": [float(x) for x in theta_this.tolist()],
                "phi":   [float(x) for x in phi_this.tolist()],
                "theta_center": float(theta_c),
                "phi_center": float(phi_c),
                "n_stars": int(len(idxs))
            }

            if solved:
                payload.update({
                    "err": float(err_deg),
                    "n_matches": int(n_match),
                    "WCS": wcs_to_header_dict(wcs)
                })
                solved_mask[(k, proj_type)] = True
            if not solved:
                payload.update({
                    "err": float("nan"),
                    "n_matches": 0
                    # WCS omitted for now; may be filled by neighbors later
                })
                solved_mask[(k, proj_type)] = False

            node[proj_type] = payload

    blob = {"meta": meta, "result": results}

    # Write JSON atomically
    tmp_path = json_out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_out_path)
    return json_out_path

def make_ra_dec_query_from_json(json_path: str):
    """
    Loads a saved JSON solution and returns a function:

        query(x_vec, y_vec) -> (ra_deg, dec_deg, err_deg)

    For each query point:
      - Convert (x,y) → (r,phi) around image center
      - For each projection type:
          r→theta, pick healpix tile
          rotate (theta,phi) to the tile's local frame (same as solve)
          (theta',phi') → (r',phi') → (u',v')
          feed (u',v') into the tile WCS to get (RA,Dec)
      - Use the first projection that resolves successfully
    """
    with open(json_path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    meta = blob.get("meta", {})
    results = blob.get("result", {})

    CCAM = meta.get("CENTER", [np.nan, np.nan])
    R = meta.get("R", np.nan)
    nside = meta.get("NSIDE", 4)
    proj_types_present: List[str] = meta.get(
        "PROJECTION_TYPES",
        list(results[next(iter(results))].keys()) if results else ['TAN', 'STG', 'ARC', 'ZEA', 'SIN']
    )

    # Preconstruct WCS objects and cache tile centers for all tiles
    wcs_cache: Dict[Tuple[int, str], astropy.wcs.WCS] = {}
    err_cache: Dict[Tuple[int, str], float] = {}
    center_cache: Dict[Tuple[int, str], Tuple[float, float]] = {}

    for k_str, per_proj in results.items():
        hpix = int(k_str)
        for proj_type, payload in per_proj.items():
            # Cache tile center (needed to rotate query points into the local frame)
            theta_c = payload.get("theta_center", None)
            phi_c = payload.get("phi_center", None)
            if theta_c is not None and phi_c is not None:
                center_cache[(hpix, proj_type)] = (float(theta_c), float(phi_c))

            if "WCS" not in payload:
                continue
            try:
                w = header_dict_to_wcs(payload["WCS"])
                wcs_cache[(hpix, proj_type)] = w
                err_cache[(hpix, proj_type)] = float(payload.get("err", float("nan")))
            except Exception:
                # Skip malformed WCS entries
                continue

    CCAM_u0, CCAM_v0 = float(CCAM[0]), float(CCAM[1])
    f0 = float(R)

    def _theta_ok(theta_val: float) -> bool:
        return np.isfinite(theta_val) and (0.0 <= theta_val <= np.pi)

    def query(x_vec: np.ndarray, y_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_vec = np.asarray(x_vec, dtype=float)
        y_vec = np.asarray(y_vec, dtype=float)
        assert x_vec.shape == y_vec.shape, "x_vec and y_vec must have the same shape"

        n = x_vec.size
        ra_out = np.full(n, np.nan, dtype=float)
        dec_out = np.full(n, np.nan, dtype=float)
        err_out = np.full(n, np.nan, dtype=float)

        for i in range(n):
            x = x_vec[i]
            y = y_vec[i]

            # Convert to polar around camera center
            r_xy, phi_xy = uv2rphi(np.array([x]), np.array([y]), CCAM_u0, CCAM_v0)
            r_xy = float(r_xy[0])
            phi_xy = float(phi_xy[0])

            hit = False
            with np.errstate(invalid='ignore'):
                for proj_type in proj_types_present:
                    # 1) r → theta for this projection
                    try:
                        theta_val = float(r2theta(np.array([r_xy]), f0, proj_type)[0])
                    except Exception:
                        continue
                    if not _theta_ok(theta_val) or not np.isfinite(phi_xy):
                        continue

                    # 2) find hpix tile
                    try:
                        hpix = int(hp.ang2pix(nside, theta_val, phi_xy))
                    except Exception:
                        continue

                    key = (hpix, proj_type)
                    if key not in wcs_cache or key not in center_cache:
                        continue

                    # 3) rotate (theta,phi) into tile-local frame (same as during solve)
                    theta_c, phi_c = center_cache[key]
                    # Spherical to 3D
                    xyz = hp.ang2vec(theta_val, phi_xy)[None, :]  # shape (1,3)

                    Rz = np.array([[np.cos(-phi_c), -np.sin(-phi_c), 0],
                                   [np.sin(-phi_c),  np.cos(-phi_c), 0],
                                   [0,                0,              1]])
                    Ry = np.array([[np.cos(-theta_c), 0, np.sin(-theta_c)],
                                   [0,                 1, 0],
                                   [-np.sin(-theta_c), 0, np.cos(-theta_c)]])
                    R_mat = Ry @ Rz
                    xyz_rot = xyz @ R_mat.T  # (1,3)
                    rotated_theta, rotated_phi = hp.vec2ang(xyz_rot)

                    # 4) local (theta',phi') → (r',phi') → (u',v') in original pixel frame
                    rotated_r = float(theta2r(np.array([rotated_theta]), f0, proj_type)[0])
                    u_prime, v_prime = rphi2uv(np.array([rotated_r]), np.array([rotated_phi]), CCAM_u0, CCAM_v0)
                    u_prime = float(u_prime[0])
                    v_prime = float(v_prime[0])

                    # 5) feed rotated pixel into tile WCS
                    try:
                        w = wcs_cache[key]
                        ra, dec = w.wcs_pix2world([[u_prime, v_prime]], 0)[0]
                        ra_out[i] = float(ra)
                        dec_out[i] = float(dec)
                        err_out[i] = float(err_cache.get(key, float("nan")))
                        hit = True
                        break  # use first successful projection/tile
                    except Exception:
                        continue

            # If no hit, values remain NaN

        return ra_out, dec_out, err_out

    return query
# -----------------------------
# Example CLI usage
# -----------------------------
if __name__ == "__main__":
    # Run the pipeline and write JSON
    for INPUT_JPG in INPUT_IMG_LIST:
        print(f"Processing image: {INPUT_JPG}")

        base = os.path.splitext(os.path.basename(INPUT_JPG))[0]
        json_out_path = os.path.join(TEMP_DIR, f"{base}_astrometry.json")
        # skip if already exists
        if os.path.exists(json_out_path):
            print(f"  JSON output already exists at {json_out_path}, skipping.")
            continue
        json_path = pipeline(
            image_path=INPUT_JPG,
            input_catalog=REF_CAT_CSV,
            SITE_LAT=SITE_LAT,
            SITE_LON=SITE_LON,
            SITE_ALT=SITE_ALT,
            CCAM=INIT_CENTER,
            R=INIT_RADIUS,
            N_SIGMA_DETECTION=N_SIGMA_DETECTION,
            N_PIXEL_AREA=N_PIXEL_AREA,
            proportion=VALID_STAR_POS_PROPORTION,
            nside=NSIDE,
            json_out_path=None  # put file under TEMP_DIR with derived name
        )
        print(f"Wrote astrometry JSON to: {json_path}")
    #     img_name = os.path.basename(INPUT_JPG).split('.')[0]
    #     json_path = f"/mnt/hgfs/cloud/calibration/temp/{img_name}_astrometry.json"
    #     # Build a query function from the saved JSON
    #     query_ra_dec = make_ra_dec_query_from_json(json_path)

    #     # Example: query a few points (vectorized)
    #     xq = np.array([INIT_CENTER[0], INIT_CENTER[0] + 100, INIT_CENTER[0] - 200], dtype=float)
    #     yq = np.array([INIT_CENTER[1], INIT_CENTER[1] + 50,  INIT_CENTER[1] + 75], dtype=float)

    #     xq = 32 + np.arange(0, 4096, 64)
    #     yq = 32 + np.arange(0, 4096, 64)
    #     xq, yq = np.meshgrid(xq, yq)
    #     xq = xq.ravel()
    #     yq = yq.ravel()
    #     ra, dec, err = query_ra_dec(xq, yq)
    #     # for i in range(len(xq)):
    #     #     print(f"(x={xq[i]:.1f}, y={yq[i]:.1f}) -> RA={ra[i]:.6f} deg, Dec={dec[i]:.6f} deg, err={err[i]}")
    #     # calculate az alt for each point for ra, dec
    #     timestamp_str = os.path.basename(INPUT_JPG).split('.')[0]
    #     obstime, location, altaz_frame = observation_conditions(
    #     timestamp_str, SITE_LAT, SITE_LON, SITE_ALT)
    #     skycoords = astropy.coordinates.SkyCoord(ra=ra*units.deg, dec=dec*units.deg, frame='icrs')
    #     altaz = skycoords.transform_to(altaz_frame)
    #     az = altaz.az.deg
    #     alt = altaz.alt.deg

    #     # Visualize the results
    #     import matplotlib.pyplot as plt
    #     from matplotlib.colors import Normalize
    #     if not plt.get_fignums():
    #         fig, ax = plt.subplots(figsize=(10, 10))

    #         im = ax.imshow(load_image(INPUT_JPG), cmap='gray',vmin=0,vmax=50)#, origin='lower', vmin=0, vmax=255)

    #         # get the range of err
    #         # err_min = np.nanmin(dec)
    #         # err_max = np.nanmax(dec)
    #         # # avoid zero range
    #         # if not np.isfinite(err_min) or not np.isfinite(err_max) or err_max == err_min:
    #         #     err_max = err_min + 1e-9

    #         norm = Normalize(vmin=0, vmax=90)
    #         cmap = plt.cm.viridis
    #         sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #         sm.set_array([])  # or np.array([]); required by some Matplotlib versions
    #         fig.colorbar(sm, ax=ax, label='alt (degrees)')
    #         ax.set_title("Queried points on image")
    #         ax.set_xlabel("X (pixels)")
    #         ax.set_ylabel("Y (pixels)")
    #         ax.set_xlim(0, 4096)
    #         ax.set_ylim(0, 4096)
    #         ax.grid(color='white', linestyle='--', alpha=0.5)
    #     # draw 64x64 squares colored by alt
    #     box_size = 64
    #     for i in range(len(xq)):
    #         if np.isfinite(alt[i]):
    #             rect = plt.Rectangle(
    #                 (xq[i]-box_size/2, yq[i]-box_size/2),
    #                 box_size, box_size,
    #                 linewidth=1, edgecolor='none',
    #                 facecolor=cmap(norm(alt[i]))
    #             )
    #             ax.add_patch(rect)



    #     # colorbar for err
        

    # plt.show()
