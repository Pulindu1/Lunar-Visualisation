"""
terrain_explorer.py  --  Artemis III Technical Terrain Explorer

A mission-planning tool for the Moon's south pole: multi-scale maps, 3D
displacement rendering, isoline analysis, and click-to-sample elevation,
illumination, slope and safety readouts.

Run:
    python3 src/terrain_explorer.py

Opens a browser at http://localhost:5006 automatically.
"""

import os
import sys
import threading
import tempfile
import warnings
import numpy as np
from PIL import Image

import panel as pn
import param

from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, TapTool, CrosshairTool,
    LinearColorMapper, ColorBar, FixedTicker, LabelSet,
    HoverTool,
)
from bokeh.models.callbacks import CustomJS
from bokeh.events import Tap

import xarray as xr
import pygmt
from scipy.ndimage import zoom as scipy_zoom
import matplotlib
matplotlib.use("agg")   # non-interactive backend; required for worker-thread rendering
import matplotlib.pyplot as plt
import plotly.graph_objects as go

try:
    import cmcrameri.cm as cmc
    _HAS_CMC = True
except ImportError:
    _HAS_CMC = False

warnings.filterwarnings("ignore")

# Data lives in dataset/ at the project root, one level above this file, so the
# app can be launched from any working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEIGHTMAP_PATH    = os.path.join(PROJECT_ROOT, "dataset", "heightmaps",   "ldem_16_uint.tif")
ILLUMINATION_PATH = os.path.join(PROJECT_ROOT, "dataset", "illumination", "lroc_color_poles_4k.tif")

LON_MIN, LON_MAX =   0.0, 360.0
LAT_MAX, LAT_MIN =  90.0, -90.0

ELEV_MIN_M = -9000.0
ELEV_MAX_M =  7000.0

SLOPE_DANGER_DEG = 15.0

# 16 px/degree; m_per_deg = (2*pi*1,737,400)/360 = 30,341 m; m_per_px = 30,341/16 ≈ 1896 m
GROUND_RES_M_PER_PX = 1900.0

REGIONS = {
    "Full south pole":            [1,   359, -90, -60],
    "Shackleton crater":          [100, 220, -90, -88],
    "Peak near Shackleton":       [110, 200, -90, -87],
    "Connecting Ridge":           [120, 200, -90, -87],
    "Connecting Ridge Extended":  [100, 210, -90, -86],
    "de Gerlache Rim 1":          [260, 340, -90, -87],
    "de Gerlache Rim 2":          [280, 360, -90, -87],
    "de Gerlache-Kocher Massif":  [240, 340, -89, -85],
    "Haworth":                    [0,    60, -90, -87],
    "Malapert Massif":            [0,    60, -87, -83],
    "Leibnitz Beta Plateau":      [140, 220, -87, -83],
    "Nobile Rim 1":               [40,  120, -90, -87],
    "Nobile Rim 2":               [60,  140, -90, -87],
    "Amundsen / Faustini Rim":    [60,  130, -89, -85],
}

# Spans -90° to -80°: covers all 13 Artemis candidate sites
POLAR_DISC_REGION = [0, 360, -90, -80]

REGION_VERT_EXAG = {
    "Full south pole":            "2c",
    "Shackleton crater":          "4c",
    "Peak near Shackleton":       "4c",
    "Connecting Ridge":           "4c",
    "Connecting Ridge Extended":  "3c",
    "de Gerlache Rim 1":          "4c",
    "de Gerlache Rim 2":          "4c",
    "de Gerlache-Kocher Massif":  "3c",
    "Haworth":                    "4c",
    "Malapert Massif":            "3c",
    "Leibnitz Beta Plateau":      "3c",
    "Nobile Rim 1":               "4c",
    "Nobile Rim 2":               "4c",
    "Amundsen / Faustini Rim":    "3c",
}

CMAP_OPTIONS = ["batlow", "batlowS", "roma", "oleron", "lapaz", "hawaii"]

_CANDIDATE_PINS = []

ARTEMIS_SITE_CENTRES = {
    "Shackleton crater":         [160, -89.9],
    "Peak near Shackleton":      [155, -89.0],
    "Connecting Ridge":          [160, -88.8],
    "Connecting Ridge Extended": [155, -88.5],
    "de Gerlache Rim 1":         [300, -88.7],
    "de Gerlache Rim 2":         [320, -88.7],
    "de Gerlache-Kocher Massif": [290, -87.0],
    "Haworth":                   [ 20, -88.5],
    "Malapert Massif":           [ 15, -85.5],
    "Leibnitz Beta Plateau":     [180, -85.5],
    "Nobile Rim 1":              [ 75, -88.8],
    "Nobile Rim 2":              [100, -88.8],
    "Amundsen / Faustini Rim":   [ 95, -87.0],
}


class DataManager:
    """Loads, calibrates, aligns, and computes derived layers from the heightmap and illumination datasets."""

    def __init__(self):
        print("[DataManager] Loading heightmap ...", flush=True)
        self.hmap_m = self._load_heightmap()

        print("[DataManager] Loading illumination ...", flush=True)
        self.illum_rgb = self._load_illumination()
        self.illum_lum = self._to_luminance(self.illum_rgb)

        print("[DataManager] Aligning resolutions ...", flush=True)
        target = self.illum_lum.shape
        if self.hmap_m.shape != target:
            zy = target[0] / self.hmap_m.shape[0]
            zx = target[1] / self.hmap_m.shape[1]
            self.hmap_m = scipy_zoom(self.hmap_m, (zy, zx), order=3)

        print("[DataManager] Computing derived layers ...", flush=True)
        self.slope_deg = self._compute_slope(self.hmap_m)
        self.psr_mask  = self._compute_psr(self.illum_lum, threshold=50)
        self.safety    = self._compute_safety()

        self.H, self.W = self.hmap_m.shape
        print(f"[DataManager] Ready. Grid: {self.W}x{self.H}", flush=True)

    def _load_heightmap(self):
        """Load ldem_16_uint.tif and return elevation in metres."""
        arr = np.array(Image.open(HEIGHTMAP_PATH))
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            elev_m = arr.astype(np.float32) * 1000.0
        elif arr.dtype in (np.uint16, np.int16):
            # 0.5 m/DN, offset so 0 DN = -8000 m
            elev_m = arr.astype(np.float32) * 0.5 - 8000.0
        else:
            elev_m = (arr.astype(np.float32) / 255.0
                      * (ELEV_MAX_M - ELEV_MIN_M) + ELEV_MIN_M)
        return elev_m

    def _load_illumination(self):
        img = Image.open(ILLUMINATION_PATH).convert("RGB")
        return np.array(img).astype(np.float32)

    def _to_luminance(self, rgb):
        # ITU-R BT.601 perceptual luminance coefficients
        return (0.299 * rgb[..., 0]
              + 0.587 * rgb[..., 1]
              + 0.114 * rgb[..., 2]).astype(np.float32)

    def _compute_slope(self, elev_m):
        """Compute slope in degrees from elevation gradient."""
        gy, gx = np.gradient(elev_m)
        slope_mpm = np.sqrt(gx**2 + gy**2) / GROUND_RES_M_PER_PX
        return np.degrees(np.arctan(slope_mpm)).astype(np.float32)

    def _compute_psr(self, lum, threshold=50):
        """Threshold illumination to derive Permanently Shadowed Regions."""
        return (lum < threshold).astype(bool)

    def _compute_safety(self):
        """Composite landing safety score (0=unsafe, 1=ideal) from slope, illumination, and elevation."""
        flatness  = np.clip(1.0 - self.slope_deg / SLOPE_DANGER_DEG, 0, 1)
        sunlight  = self.illum_lum / 255.0
        elev_norm = np.clip(
            (self.hmap_m - ELEV_MIN_M) / (ELEV_MAX_M - ELEV_MIN_M), 0, 1)
        elev_score = 1.0 - np.abs(elev_norm - 0.5) * 2
        return (0.4 * flatness + 0.4 * sunlight + 0.2 * elev_score
                ).astype(np.float32)

    def sample_point(self, lon, lat):
        """Return data values at a geographic coordinate (degrees)."""
        py = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * self.H)
        px = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * self.W)
        py = int(np.clip(py, 0, self.H - 1))
        px = int(np.clip(px, 0, self.W - 1))
        return {
            "lon":          lon,
            "lat":          lat,
            "elevation_m":  float(self.hmap_m[py, px]),
            "illumination": float(self.illum_lum[py, px] / 255.0 * 100.0),
            "slope_deg":    float(self.slope_deg[py, px]),
            "safety_score": float(self.safety[py, px]),
            "is_psr":       bool(self.psr_mask[py, px]),
        }

    def crop(self, region):
        """Crop all arrays to a geographic bounding box [lon0, lon1, lat0, lat1]."""
        lon0, lon1, lat0, lat1 = region
        r0 = int((LAT_MAX - lat1) / (LAT_MAX - LAT_MIN) * self.H)
        r1 = int((LAT_MAX - lat0) / (LAT_MAX - LAT_MIN) * self.H)
        c0 = int((lon0 - LON_MIN)  / (LON_MAX - LON_MIN) * self.W)
        c1 = int((lon1 - LON_MIN)  / (LON_MAX - LON_MIN) * self.W)
        r0, r1 = max(r0, 0), min(r1, self.H)
        c0, c1 = max(c0, 0), min(c1, self.W)
        # Handle lon wrapping for full-pole view (0-360)
        if c0 >= c1:
            c0, c1 = 0, self.W
        sl = (slice(r0, r1), slice(c0, c1))

        raw = {
            "hmap_m":    self.hmap_m[sl],
            "illum_lum": self.illum_lum[sl],
            "slope_deg": self.slope_deg[sl],
            "psr_mask":  self.psr_mask[sl],
            "safety":    self.safety[sl],
        }

        # Polar crops near the south pole have very few rows; upsample to at
        # least MIN_ROWS to avoid stripe artefacts in renderers.
        MIN_ROWS = 200
        rows_raw = raw["hmap_m"].shape[0]
        if rows_raw < MIN_ROWS:
            scale = MIN_ROWS / rows_raw
            upsampled = {}
            for key, arr in raw.items():
                if key == "psr_mask":
                    upsampled[key] = scipy_zoom(
                        arr.astype(np.float32), scale, order=0
                    ).astype(bool)
                else:
                    upsampled[key] = scipy_zoom(
                        arr.astype(np.float32), scale, order=3
                    ).astype(arr.dtype)
            return upsampled

        return raw

    def to_xarray(self, arr2d, region, name="elevation", units="m"):
        """Wrap a cropped numpy array as xarray.DataArray for PyGMT."""
        lon0, lon1, lat0, lat1 = region
        rows, cols = arr2d.shape
        lat_coords = np.linspace(lat1, lat0, rows)
        lon_coords = np.linspace(lon0, lon1, cols)
        return xr.DataArray(
            arr2d,
            dims=["lat", "lon"],
            coords={
                "lat": lat_coords,
                "lon": lon_coords,
            },
            attrs={
                "long_name": name,
                "units": units,
                # Explicit spacing prevents GMT from inferring a wrong increment from a coarse crop
                "spacing": float((lon1 - lon0) / max(cols - 1, 1)),
            },
        )


def _get_cmap(name):
    """Return a matplotlib colormap, preferring cmcrameri if installed."""
    if _HAS_CMC:
        try:
            return getattr(cmc, name)
        except AttributeError:
            pass
    fallbacks = {
        "batlow":  "viridis",
        "batlowS": "viridis",
        "roma":    "RdBu_r",
        "oleron":  "terrain",
        "lapaz":   "plasma",
        "hawaii":  "magma",
    }
    return plt.get_cmap(fallbacks.get(name, "viridis"))


def array_to_rgba(arr2d, cmap_name="batlow", vmin=None, vmax=None,
                  flip_y=True):
    """Convert a 2D float array to uint32 packed RGBA for Bokeh image_rgba (little-endian R|G<<8|B<<16|A<<24)."""
    vmin = vmin if vmin is not None else float(arr2d.min())
    vmax = vmax if vmax is not None else float(arr2d.max())
    norm  = np.clip((arr2d - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    cm    = _get_cmap(cmap_name)
    rgba_f  = cm(norm)
    rgba_u8 = (rgba_f * 255).astype(np.uint8)
    packed  = (  rgba_u8[..., 0].astype(np.uint32)
              | (rgba_u8[..., 1].astype(np.uint32) << 8)
              | (rgba_u8[..., 2].astype(np.uint32) << 16)
              | (rgba_u8[..., 3].astype(np.uint32) << 24))
    if flip_y:
        packed = np.flipud(packed)
    return packed


def apply_overlay(base_rgba_u32, crop, show_psr, show_slope, show_safety,
                  psr_alpha=0.72, psr_threshold=50,
                  show_solar_mask=False, solar_mask_pct=70,
                  show_golden_zone=False):
    """Blend overlay layers (PSR, slope danger, safety score) onto a packed uint32 RGBA Bokeh image."""
    h, w = base_rgba_u32.shape

    def _resize(arr):
        if arr.shape != (h, w):
            zy = h / arr.shape[0]
            zx = w / arr.shape[1]
            arr = scipy_zoom(arr.astype(float), (zy, zx), order=3)
        return arr

    # Unpack uint32 RGBA -> float channels [0,1]
    r = ( base_rgba_u32        & 0xFF).astype(float) / 255.0
    g = ((base_rgba_u32 >>  8) & 0xFF).astype(float) / 255.0
    b = ((base_rgba_u32 >> 16) & 0xFF).astype(float) / 255.0

    if show_psr:
        illum_raw = _resize(crop["illum_lum"].astype(float))
        psr = illum_raw < psr_threshold
        alpha = psr_alpha
        r = np.where(psr, r * (1 - alpha) + 0.000 * alpha, r)   # #00e5ff R=0
        g = np.where(psr, g * (1 - alpha) + 0.898 * alpha, g)   # #00e5ff G=229
        b = np.where(psr, b * (1 - alpha) + 1.000 * alpha, b)   # #00e5ff B=255

    if show_slope:
        sl = _resize(crop["slope_deg"])
        warn   = (sl > 10) & (sl <= 15)
        danger =  sl > 15
        aw = 0.55
        r = np.where(warn, r * (1-aw) + 1.000 * aw, r)
        g = np.where(warn, g * (1-aw) + 0.647 * aw, g)
        b = np.where(warn, b * (1-aw) + 0.000 * aw, b)
        ad = 0.65
        r = np.where(danger, r * (1-ad) + 0.863 * ad, r)
        g = np.where(danger, g * (1-ad) + 0.118 * ad, g)
        b = np.where(danger, b * (1-ad) + 0.118 * ad, b)

    if show_safety:
        sc  = _resize(crop["safety"])
        top = sc > 0.80
        s   = 0.45
        r = np.where(top, r * (1-s) + 0.196 * s, r)
        g = np.where(top, g * (1-s) + 0.824 * s, g)
        b = np.where(top, b * (1-s) + 0.314 * s, b)

    if show_solar_mask:
        illum = _resize(crop["illum_lum"].astype(float))
        illum_pct = illum / 255.0 * 100.0
        mask = illum_pct >= solar_mask_pct
        am = 0.55
        r = np.where(mask, r * (1 - am) + 0.980 * am, r)   # #facc15 R=250
        g = np.where(mask, g * (1 - am) + 0.800 * am, g)   # #facc15 G=204
        b = np.where(mask, b * (1 - am) + 0.082 * am, b)   # #facc15 B=21

    if show_golden_zone:
        illum_gz = _resize(crop["illum_lum"].astype(float))
        illum_pct_gz = illum_gz / 255.0 * 100.0
        sl_gz = _resize(crop["slope_deg"])
        golden = (illum_pct_gz >= solar_mask_pct) & (sl_gz < SLOPE_DANGER_DEG)
        ag = 0.65
        r = np.where(golden, r * (1 - ag) + 0.851 * ag, r)  # #d97706 R=217
        g = np.where(golden, g * (1 - ag) + 0.467 * ag, g)  # #d97706 G=119
        b = np.where(golden, b * (1 - ag) + 0.024 * ag, b)  # #d97706 B=6

    r, g, b = (np.clip(x * 255, 0, 255).astype(np.uint32) for x in (r, g, b))
    return r | (g << 8) | (b << 16) | (255 << 24)


def extract_contour_lines(arr2d, region, levels, extract_labels=False):
    """Extract contour line coordinates from a 2D elevation array using matplotlib's marching squares."""
    lon0, lon1, lat0, lat1 = region

    rows, cols = arr2d.shape
    if cols > 512:
        factor = cols // 512
        arr2d = arr2d[::factor, ::factor]
        rows, cols = arr2d.shape

    lons = np.linspace(lon0, lon1, cols)
    lats = np.linspace(lat1, lat0, rows)

    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(lons, lats, arr2d, levels=levels)

    xs, ys = [], []
    for segs in cs.allsegs:
        for path in segs:
            if len(path) > 1:
                xs.append(path[:, 0].tolist())
                ys.append(path[:, 1].tolist())

    if not extract_labels:
        plt.close(fig_tmp)
        return xs, ys

    label_objs = ax_tmp.clabel(cs, inline=True, fontsize=7, fmt="%d m")
    plt.close(fig_tmp)

    label_lons  = [t.get_position()[0] for t in label_objs]
    label_lats  = [t.get_position()[1] for t in label_objs]
    label_texts = [t.get_text() for t in label_objs]

    return xs, ys, label_lons, label_lats, label_texts


def _bokeh_palette(cmap_name, n=256):
    """Convert a matplotlib/cmcrameri colormap to a Bokeh hex palette list."""
    cm = _get_cmap(cmap_name)
    return [
        "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        for r, g, b, _ in (cm(i / (n - 1)) for i in range(n))
    ]


def _physical_aspect_y(lon_span, lat_span, lat0, lat1):
    """Compute y:x aspect ratio for the Plotly 3D scene using physical lunar surface metres."""
    import math
    LUNAR_RADIUS_M = 1_737_400.0
    DEG_TO_RAD     = math.pi / 180.0

    m_per_deg_lat = LUNAR_RADIUS_M * DEG_TO_RAD
    mean_lat_rad  = abs((lat0 + lat1) / 2.0) * DEG_TO_RAD
    m_per_deg_lon = LUNAR_RADIUS_M * math.cos(mean_lat_rad) * DEG_TO_RAD
    m_per_deg_lon = max(m_per_deg_lon, 1.0)

    ns_metres = lat_span * m_per_deg_lat
    ew_metres = lon_span * m_per_deg_lon

    ratio = ns_metres / max(ew_metres, 1.0)
    return float(np.clip(ratio, 0.15, 6.0))


def _physical_figure_height(region, target_width_px=700,
                             min_height=200, max_height=700):
    """Compute Bokeh figure height so one pixel represents equal physical distance on both axes."""
    import math
    lon0, lon1, lat0, lat1 = region
    LUNAR_R = 1_737_400.0
    DEG2RAD = math.pi / 180.0

    lon_span = max(lon1 - lon0, 1e-3)
    lat_span = max(lat1 - lat0, 1e-3)

    mean_lat_rad  = abs((lat0 + lat1) / 2.0) * DEG2RAD
    m_per_deg_lon = LUNAR_R * math.cos(mean_lat_rad) * DEG2RAD
    m_per_deg_lat = LUNAR_R * DEG2RAD

    ew_m = lon_span * max(m_per_deg_lon, 1.0)
    ns_m = lat_span * m_per_deg_lat

    phys_ratio = ns_m / max(ew_m, 1.0)
    height_px  = int(round(target_width_px * phys_ratio))
    return int(np.clip(height_px, min_height, max_height))


def _lonlat_to_disc(lons_deg, lats_deg, r_max=10.0, lat_min=-90.0,
                    lat_max=-80.0):
    """
    Convert (longitude, latitude) to flat polar disc coordinates using azimuthal equidistant projection.

    r = (lat - lat_min)/(lat_max - lat_min)*r_max; theta = lon*pi/180; x = r*sin(theta); y = r*cos(theta).
    """
    import math
    lons = np.asarray(lons_deg, dtype=float)
    lats = np.asarray(lats_deg, dtype=float)
    r     = (lats - lat_min) / (lat_max - lat_min) * r_max
    theta = lons * math.pi / 180.0
    xs    = r * np.sin(theta)
    ys    = r * np.cos(theta)
    return xs, ys


def _disc_to_lonlat(x, y, r_max=10.0, lat_min=-90.0, lat_max=-80.0):
    """
    Back-transform flat polar disc coordinates to (longitude, latitude).

    Returns None if the click is outside the disc (r > r_max).
    """
    import math
    r = math.sqrt(x * x + y * y)
    if r > r_max:
        return None
    lat = lat_min + (r / r_max) * (lat_max - lat_min)
    # atan2(x, y) measures azimuth from north (+y), clockwise
    theta = math.atan2(x, y)
    lon   = math.degrees(theta) % 360.0
    return lon, lat


def _reproject_image_to_disc(img_rgba_u32, region, disc_px=700,
                               r_max=10.0, lat_min=-90.0, lat_max=-80.0):
    """
    Remap an equirectangular RGBA image into a polar disc pixel grid using bilinear interpolation.

    Pixels outside the disc or source image are set to transparent.
    """
    from scipy.ndimage import map_coordinates
    lon0, lon1, lat0, lat1 = region
    H, W = img_rgba_u32.shape

    r_ch = ( img_rgba_u32        & 0xFF).astype(np.float32)
    g_ch = ((img_rgba_u32 >>  8) & 0xFF).astype(np.float32)
    b_ch = ((img_rgba_u32 >> 16) & 0xFF).astype(np.float32)
    a_ch = ((img_rgba_u32 >> 24) & 0xFF).astype(np.float32)

    # img_rgba_u32 was produced with flip_y=True: row 0 = lat0 (south)
    half = disc_px / 2.0
    py_idx, px_idx = np.mgrid[0:disc_px, 0:disc_px]
    # Convert pixel indices to plot-unit disc coordinates
    px_plot = (px_idx - half) / half * r_max
    py_plot = (py_idx - half) / half * r_max

    r_sq   = px_plot**2 + py_plot**2
    inside = r_sq <= r_max**2

    r_val  = np.where(inside, np.sqrt(r_sq), 0.0)
    lat    = lat_min + (r_val / r_max) * (lat_max - lat_min)

    theta  = np.arctan2(px_plot, py_plot)       # azimuth from north
    lon    = np.degrees(theta) % 360.0          # normalise to [0, 360)

    # Source row/col indices; row 0 = lat0 because flip_y was applied upstream
    src_col = (lon  - lon0) / (lon1 - lon0) * (W - 1)
    src_row = (lat  - lat0) / (lat1 - lat0) * (H - 1)

    src_col = np.clip(src_col, 0, W - 1)
    src_row = np.clip(src_row, 0, H - 1)

    coords = [src_row.ravel(), src_col.ravel()]
    out_r = map_coordinates(r_ch, coords, order=1, mode="nearest").reshape(disc_px, disc_px)
    out_g = map_coordinates(g_ch, coords, order=1, mode="nearest").reshape(disc_px, disc_px)
    out_b = map_coordinates(b_ch, coords, order=1, mode="nearest").reshape(disc_px, disc_px)
    out_a = map_coordinates(a_ch, coords, order=1, mode="nearest").reshape(disc_px, disc_px)

    out_r = np.where(inside, out_r, 0.0)
    out_g = np.where(inside, out_g, 0.0)
    out_b = np.where(inside, out_b, 0.0)
    out_a = np.where(inside, out_a, 0.0)

    out = (  out_r.astype(np.uint32)
           | (out_g.astype(np.uint32) <<  8)
           | (out_b.astype(np.uint32) << 16)
           | (out_a.astype(np.uint32) << 24))
    return out


def _transform_contour_lines(xs, ys, r_max=10.0, lat_min=-90.0,
                               lat_max=-80.0):
    """Transform contour line segments from (lon, lat) degrees to polar disc plot coordinates."""
    xs_disc, ys_disc = [], []
    for seg_x, seg_y in zip(xs, ys):
        dx, dy = _lonlat_to_disc(seg_x, seg_y, r_max=r_max,
                                  lat_min=lat_min, lat_max=lat_max)
        xs_disc.append(dx.tolist())
        ys_disc.append(dy.tolist())
    return xs_disc, ys_disc


class RenderEngine:
    """Produces static PNG files from PyGMT for 3D perspective and isoline visualisation modes."""

    @staticmethod
    def _pick_projection(region):
        """Select GMT projection: azimuthal equidistant for wide regions (lon_span > 90), Mercator otherwise."""
        lon0, lon1, *_ = region
        lon_span = lon1 - lon0
        if lon_span > 90:
            return "E0/-90/15c"
        return "M15c"

    @staticmethod
    def render_3d(grid_xr, region, azimuth=130, elev_angle=30,
                  zsize="2c", cmap="SCM/batlow", drape_xr=None):
        """Render 3D perspective terrain using grdview with optional illumination drape."""
        lon0, lon1, lat0, lat1 = region
        # Clamp away from exact poles (-90/+90 are degenerate for many GMT projections)
        lat0 = max(lat0, -89.9)
        lat1 = min(lat1,  89.9)
        vmin = float(np.nanmin(grid_xr.values))
        vmax = float(np.nanmax(grid_xr.values))
        tmp  = tempfile.NamedTemporaryFile(suffix="_3d.png", delete=False)
        out  = tmp.name
        tmp.close()

        inc = max((vmax - vmin) / 100.0, 1.0)

        lon_span = lon1 - lon0
        proj = RenderEngine._pick_projection(region)
        tick_x = "45" if lon_span > 90 else "10"
        tick_y = "10"

        fig = pygmt.Figure()

        if drape_xr is not None:
            pygmt.makecpt(cmap="gray", series=[0, 255, 2.55], continuous=True)
        else:
            pygmt.makecpt(cmap=cmap, series=[vmin, vmax, inc], continuous=True)

        # +a315: NW azimuth (cartographic convention); +ne0.6: Laplace normalisation amplitude 0.6
        _shading_str = "+a315+ne0.6"

        # surftype="i600": image plot at 600 DPI avoids blocky polygon fill at high zoom
        _grdview_kwargs = dict(
            grid=grid_xr,
            region=[lon0, lon1, lat0, lat1],
            projection=proj,
            perspective=[azimuth, elev_angle],
            zsize=zsize,
            surftype="i600",
            cmap=True,
            shading=_shading_str,
            frame=[f"xa{tick_x}", f"ya{tick_y}", "WSnE"],
        )
        if drape_xr is not None:
            _grdview_kwargs["drapegrid"] = drape_xr

        fig.grdview(**_grdview_kwargs)

        if drape_xr is not None:
            fig.colorbar(
                frame=["a50", "x+lIllumination", "y+l(luminance)"],
                position="JBC+o0c/-1.5c+w10c/0.4c+h",
            )
        else:
            fig.colorbar(
                frame=["a2000", "x+lElevation", "y+lm"],
                position="JBC+o0c/-1.5c+w10c/0.4c+h",
            )
        fig.savefig(out, dpi=300, crop=True)

        # Replace white background with transparency for dark Panel UI
        _img = Image.open(out).convert("RGBA")
        _pixels = _img.getdata()
        _new = []
        for r, g, b, a in _pixels:
            # batlow never produces near-white for elevation data; safe to treat as background
            if r > 240 and g > 240 and b > 240:
                _new.append((255, 255, 255, 0))
            else:
                _new.append((r, g, b, a))
        _img.putdata(_new)
        _img.save(out)

        return out

    @staticmethod
    def render_contour(grid_xr, region, contour_interval=500,
                       annotation_interval=2000, cmap="SCM/batlow"):
        """Render 2D colour-coded terrain with isoline overlay via grdimage + grdcontour."""
        lon0, lon1, lat0, lat1 = region
        lat0 = max(lat0, -89.9)
        lat1 = min(lat1,  89.9)
        vmin = float(np.nanmin(grid_xr.values))
        vmax = float(np.nanmax(grid_xr.values))
        tmp  = tempfile.NamedTemporaryFile(suffix="_contour.png", delete=False)
        out  = tmp.name
        tmp.close()

        inc = max((vmax - vmin) / 100.0, 1.0)

        lon_span = lon1 - lon0
        adaptive_annot = (annotation_interval
                          if lon_span < 180
                          else max(annotation_interval, 4000))

        fig = pygmt.Figure()
        pygmt.makecpt(cmap=cmap, series=[vmin, vmax, inc], continuous=True)
        fig.grdimage(
            grid=grid_xr,
            region=[lon0, lon1, lat0, lat1],
            projection="Q14c",
            cmap=True,
            shading="+a315+ne0.6",
            frame=True,
        )
        # Two separate grdcontour calls are needed because combining a and c pen prefixes
        # in a Python list is not reliably forwarded to GMT's -W flag in all PyGMT versions
        fig.grdcontour(
            grid=grid_xr,
            levels=float(contour_interval),
            annotation="n",
            pen="0.2p,white,--",
        )
        fig.grdcontour(
            grid=grid_xr,
            levels=float(adaptive_annot),
            annotation=f"{adaptive_annot}+f7p,white+gblack@50",
            pen="0.7p,white",
        )
        fig.colorbar(
            frame=["a1000", "x+lElevation", "y+lm"],
            position="JBC+o0/1.5c+w10c/0.4c",
        )
        fig.savefig(out, dpi=300)
        return out

    @staticmethod
    def render_polar_contour(grid_xr, region,
                             contour_interval=500,
                             annotation_interval=2000,
                             cmap="SCM/batlow"):
        """Render polar stereographic terrain map with isoline overlay centred on the south pole."""
        lon0, lon1, lat0, lat1 = region
        lat0 = max(lat0, -89.9)
        lat1 = min(lat1,  89.9)

        vmin = float(np.nanmin(grid_xr.values))
        vmax = float(np.nanmax(grid_xr.values))
        inc  = max((vmax - vmin) / 100.0, 1.0)

        tmp = tempfile.NamedTemporaryFile(suffix="_polar_contour.png",
                                          delete=False)
        out = tmp.name
        tmp.close()

        lon_span = lon1 - lon0
        adaptive_annot = (annotation_interval
                          if lon_span < 180
                          else max(annotation_interval, 4000))

        fig = pygmt.Figure()
        pygmt.makecpt(cmap=cmap, series=[vmin, vmax, inc], continuous=True)

        # S0/-90/14c: polar stereographic centred at the south pole, 14 cm diameter
        _proj = "S0/-90/14c"

        fig.grdimage(
            grid=grid_xr,
            region=[lon0, lon1, lat0, lat1],
            projection=_proj,
            cmap=True,
            shading="+a315+ne0.6",
        )

        fig.grdcontour(
            grid=grid_xr,
            region=[lon0, lon1, lat0, lat1],
            projection=_proj,
            levels=float(contour_interval),
            annotation="n",
            pen="0.2p,white,--",
        )

        fig.grdcontour(
            grid=grid_xr,
            region=[lon0, lon1, lat0, lat1],
            projection=_proj,
            levels=float(adaptive_annot),
            annotation=f"{adaptive_annot}+f7p,white+gblack@50",
            pen="0.7p,white",
        )

        fig.colorbar(
            frame=["a1000", "x+lElevation", "y+lm"],
            position="JBC+o0c/-1.5c+w10c/0.4c+h",
        )

        fig.savefig(out, dpi=300, crop=True)

        # Replace white background with transparency for dark Panel UI
        _img = Image.open(out).convert("RGBA")
        _pixels = _img.getdata()
        _new = [
            (255, 255, 255, 0) if (r > 240 and g > 240 and b > 240)
            else (r, g, b, a)
            for r, g, b, a in _pixels
        ]
        _img.putdata(_new)
        _img.save(out)

        return out

    @staticmethod
    def render_3d_plotly(crop, region, vert_exag=3.0, cmap_name="batlow",
                         show_illumination_drape=False,
                         show_threshold=False, threshold_elev_m=-2000):
        """Render an interactive 3D WebGL surface using Plotly go.Surface with optional illumination drape."""
        lon0, lon1, lat0, lat1 = region

        elev = crop["hmap_m"].astype(float)

        rows, cols = elev.shape
        if cols > 1200:
            max_cols = 400
        elif cols > 400:
            max_cols = 600
        else:
            max_cols = cols

        if cols > max_cols:
            factor = max(1, cols // max_cols)
            elev      = elev[::factor, ::factor]
            illum_lum = crop["illum_lum"][::factor, ::factor]
            psr       = crop["psr_mask"][::factor, ::factor]
            rows, cols = elev.shape
        else:
            illum_lum = crop["illum_lum"]
            psr       = crop["psr_mask"]

        lons = np.linspace(lon0, lon1, cols)
        lats = np.linspace(lat1, lat0, rows)  # lat decreases top-to-bottom

        if show_illumination_drape:
            surfacecolor   = illum_lum / 255.0
            colorscale     = [[0, "rgb(10,10,30)"], [1, "rgb(255,240,180)"]]
            cmin, cmax     = 0.0, 1.0
            cbar_title     = "Illumination"
            cbar_ticks     = [0.0, 0.25, 0.5, 0.75, 1.0]
            cbar_text      = ["0%", "25%", "50%", "75%", "100%"]
        else:
            surfacecolor = elev
            cm           = _get_cmap(cmap_name)
            n_stops      = 64
            colorscale   = [
                [i / (n_stops - 1),
                 "rgb({},{},{})".format(int(r*255), int(g*255), int(b*255))]
                for i, (r, g, b, _) in enumerate(
                    cm(j / (n_stops - 1)) for j in range(n_stops)
                )
            ]
            cmin, cmax  = float(np.nanmin(elev)), float(np.nanmax(elev))
            cbar_title  = "Elevation (m)"
            cbar_ticks  = [-8000, -6000, -4000, -2000, 0, 2000, 4000, 6000]
            cbar_text   = ["-8000 m", "-6000 m", "-4000 m", "-2000 m",
                           "0 m", "2000 m", "4000 m", "6000 m"]

        # Semi-transparent blue surface at terrain height only where PSR=True; NaN elsewhere
        psr_z = np.where(psr, elev, np.nan)

        lon_span = max(lon1 - lon0, 1e-3)
        lat_span = max(lat1 - lat0, 1e-3)

        surface = go.Surface(
            x=lons,
            y=lats,
            z=elev,
            surfacecolor=surfacecolor,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                title=dict(text=cbar_title, side="right",
                           font=dict(color="#b0b0b0", size=11)),
                tickvals=cbar_ticks,
                ticktext=cbar_text,
                len=0.75,
                thickness=14,
                bgcolor="#111827",
                tickfont=dict(color="#b0b0b0", size=9),
                outlinewidth=0,
            ),
            # ambient=0.45 prevents crater floors going completely black
            lighting=dict(
                ambient=0.45,
                diffuse=0.70,
                roughness=0.55,
                specular=0.08,
                fresnel=0.02,
            ),
            lightposition=dict(x=-1.5, y=2.0, z=3.0),
            hovertemplate=(
                "Lon: %{x:.2f} deg E<br>"
                "Lat: %{y:.2f} deg<br>"
                "Elev: %{z:.0f} m<extra></extra>"
            ),
            name="Terrain",
        )

        psr_surface = go.Surface(
            x=lons,
            y=lats,
            z=psr_z,
            colorscale=[[0, "rgba(30,80,200,0.45)"],
                        [1, "rgba(30,80,200,0.45)"]],
            showscale=False,
            hovertemplate=(
                "PSR (possible water ice)<br>"
                "Elev: %{z:.0f} m<extra></extra>"
            ),
            name="PSR",
            opacity=0.45,
            lighting=dict(ambient=1.0, diffuse=0.0),
        )

        traces = [surface, psr_surface]

        if show_threshold and threshold_elev_m is not None:
            _thr_xs, _thr_ys = extract_contour_lines(
                crop["hmap_m"],
                region,
                [threshold_elev_m],
            )
            for _seg_x, _seg_y in zip(_thr_xs, _thr_ys):
                traces.append(go.Scatter3d(
                    x=_seg_x,
                    y=_seg_y,
                    z=[float(threshold_elev_m)] * len(_seg_x),
                    mode="lines",
                    line=dict(color="#ef4444", width=5),
                    showlegend=False,
                    hoverinfo="skip",
                    name="Threshold",
                ))

        fig = go.Figure(data=traces)

        fig.update_layout(
            paper_bgcolor="#0f172a",
            scene=dict(
                xaxis=dict(
                    title=dict(text="Longitude (deg E)",
                               font=dict(color="#b0b0b0", size=10)),
                    tickfont=dict(color="#8898aa", size=8),
                    gridcolor="#1e293b",
                    backgroundcolor="#0f172a",
                    showspikes=False,
                ),
                yaxis=dict(
                    title=dict(text="Latitude (deg)",
                               font=dict(color="#b0b0b0", size=10)),
                    tickfont=dict(color="#8898aa", size=8),
                    gridcolor="#1e293b",
                    backgroundcolor="#0f172a",
                    showspikes=False,
                ),
                zaxis=dict(
                    title=dict(text="Elevation (m)",
                               font=dict(color="#b0b0b0", size=10)),
                    tickfont=dict(color="#8898aa", size=8),
                    gridcolor="#1e293b",
                    backgroundcolor="#0f172a",
                    showspikes=False,
                ),
                aspectmode="manual",
                aspectratio=dict(
                    x=1.0,
                    y=_physical_aspect_y(lon_span, lat_span, lat0, lat1),
                    z=vert_exag * 0.12,
                ),
                camera=dict(
                    eye=dict(
                        x=1.6,
                        y=-1.6,
                        # Higher z for wide tiles so the disc is visible rather than edge-on
                        z=1.4 if lon_span > 90 else 0.9,
                    ),
                    up=dict(x=0, y=0, z=1),
                ),
                bgcolor="#0f172a",
            ),
            margin=dict(l=0, r=0, t=36, b=0),
            font=dict(color="#b0b0b0"),
            height=540,
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="#111827",
                bordercolor="#1e293b",
                font=dict(color="#b0b0b0", size=10),
            ),
            title=dict(
                text=(
                    "3D Interactive Terrain  "
                    "(drag to rotate  |  scroll to zoom  |  "
                    "click to sample  |  hover for elevation)"
                ),
                font=dict(color="#94a3b8", size=12),
                x=0.01,
            ),
        )

        return fig


class AppState(param.Parameterized):
    """Central reactive state store for the Panel application."""

    region_name      = param.Selector(default="Full south pole",
                                      objects=list(REGIONS.keys()))

    contour_interval = param.Integer(default=500,  bounds=(100, 2000))
    annot_interval   = param.Integer(default=2000, bounds=(500, 5000))

    show_psr         = param.Boolean(default=True)
    show_slope       = param.Boolean(default=True)
    show_safety      = param.Boolean(default=False)

    psr_threshold    = param.Integer(default=50, bounds=(10, 120))

    cmap_name        = param.Selector(default="batlow",
                                      objects=CMAP_OPTIONS)

    show_illum_drape = param.Boolean(default=False)
    plotly_vert_exag = param.Number(default=1.0, bounds=(0.5, 10.0))

    show_threshold   = param.Boolean(default=False)
    threshold_elev_m = param.Integer(default=-2000, bounds=(-9000, 7000))

    show_illum_contour  = param.Boolean(default=False)
    illum_contour_pct   = param.Integer(default=80, bounds=(10, 100))

    show_solar_mask     = param.Boolean(default=False)
    solar_mask_pct      = param.Integer(default=70, bounds=(10, 100))

    show_golden_zone    = param.Boolean(default=False)

    show_disc_sites = param.Boolean(default=True)


def build_bokeh_map(dm, state, readout_pane, last_click):
    """Build the interactive Bokeh figure for click-to-sample and zoom."""
    region = REGIONS[state.region_name]
    lon0, lon1, lat0, lat1 = region
    crop = dm.crop(region)

    img_data = array_to_rgba(
        crop["hmap_m"], cmap_name=state.cmap_name,
        vmin=ELEV_MIN_M, vmax=ELEV_MAX_M,
    )
    img_with_overlay = apply_overlay(
        img_data, crop,
        show_psr=state.show_psr,
        show_slope=state.show_slope,
        show_safety=state.show_safety,
        psr_threshold=state.psr_threshold,
        show_solar_mask=state.show_solar_mask,
        solar_mask_pct=state.solar_mask_pct,
        show_golden_zone=state.show_golden_zone,
    )

    source = ColumnDataSource(dict(
        image=[img_with_overlay],
        x=[lon0], y=[lat0],
        dw=[lon1 - lon0], dh=[lat1 - lat0],
    ))

    _fig_h = _physical_figure_height(region, target_width_px=700)

    p = figure(
        width=700,
        height=_fig_h,
        x_range=(lon0, lon1),
        y_range=(lat0, lat1),
        x_axis_label="Longitude (°E)",
        y_axis_label="Latitude (°)",
        title="Interactive Map  [click to sample  |  scroll to zoom  |  drag to pan]",
        tools="wheel_zoom,pan,reset,save",
        active_scroll="wheel_zoom",
        match_aspect=True,
        aspect_scale=1.0,
        min_border_left=40,
        background_fill_color="#1a0b3d",
        border_fill_color="#0a0a14",
    )
    p.title.text_color          = "#e0e0e0"
    p.axis.axis_label_text_color = "#b0b0b0"
    p.axis.major_label_text_color = "#b0b0b0"
    p.image_rgba(source=source, image="image",
                 x="x", y="y", dw="dw", dh="dh")

    _mapper = LinearColorMapper(
        palette=_bokeh_palette(state.cmap_name),
        low=ELEV_MIN_M,
        high=ELEV_MAX_M,
    )
    _cbar = ColorBar(
        color_mapper=_mapper,
        ticker=FixedTicker(ticks=[-8000, -6000, -4000, -2000, 0, 2000, 4000, 6000]),
        label_standoff=8,
        border_line_color=None,
        location=(0, 0),
        title="Elevation (m)",
        title_text_color="#b0b0b0",
        title_text_font_size="11px",
        major_label_text_color="#b0b0b0",
        major_label_text_font_size="10px",
        background_fill_color="#0a0a14",
        bar_line_color=None,
        width=12,
        height=300,
    )
    p.add_layout(_cbar, "right")

    pin_source = ColumnDataSource(dict(x=[], y=[], label=[], color=[]))
    p.scatter(
        x="x", y="y", source=pin_source,
        marker="triangle", size=14,
        fill_color="color", line_color="white", line_width=1.2,
        fill_alpha=0.9,
    )
    pin_labels = LabelSet(
        x="x", y="y", text="label", source=pin_source,
        text_color="white", text_font_size="9px",
        background_fill_color="#065f46", background_fill_alpha=0.7,
        x_offset=8, y_offset=4,
    )
    p.add_layout(pin_labels)

    tap      = TapTool()
    p.add_tools(tap)
    crosshair = CrosshairTool(line_color="#ffffff", line_alpha=0.6)
    p.add_tools(crosshair)

    def on_tap(event):
        lon, lat = event.x, event.y
        res = dm.sample_point(lon, lat)
        psr_tag = "  **[PSR -- possible water ice]**" if res["is_psr"] else ""
        safety_colour = (
            "#34d399" if res["safety_score"] > 0.80 else
            "#fbbf24" if res["safety_score"] > 0.55 else
            "#f87171"
        )
        readout_pane.object = (
            f"**Lon:** {lon:.2f}°  **Lat:** {lat:.2f}°{psr_tag}\n\n"
            f"**Elevation:** {res['elevation_m']:.0f} m  &nbsp;|&nbsp;  "
            f"**Illumination:** {res['illumination']:.1f}%  &nbsp;|&nbsp;  "
            f"**Slope:** {res['slope_deg']:.1f}°  &nbsp;|&nbsp;  "
            f"**Safety Score:** "
            f"<span style='color:{safety_colour};font-weight:bold'>"
            f"{res['safety_score']:.2f}</span>"
        )
        last_click["lon"]   = lon
        last_click["lat"]   = lat
        last_click["score"] = res["safety_score"]

    p.on_event(Tap, on_tap)

    def update_image(region, crop, state):
        lon0, lon1, lat0, lat1 = region
        img_data = array_to_rgba(
            crop["hmap_m"], cmap_name=state.cmap_name,
            vmin=ELEV_MIN_M, vmax=ELEV_MAX_M,
        )
        img = apply_overlay(
            img_data, crop,
            show_psr=state.show_psr,
            show_slope=state.show_slope,
            show_safety=state.show_safety,
            psr_threshold=state.psr_threshold,
            show_solar_mask=state.show_solar_mask,
            solar_mask_pct=state.solar_mask_pct,
            show_golden_zone=state.show_golden_zone,
        )
        source.data = dict(
            image=[img],
            x=[lon0], y=[lat0],
            dw=[lon1 - lon0], dh=[lat1 - lat0],
        )
        p.x_range.start, p.x_range.end = lon0, lon1
        p.y_range.start, p.y_range.end = lat0, lat1
        p.height = _physical_figure_height(region, target_width_px=700)
        p.width  = 700

    return p, update_image, pin_source


def build_contour_map(dm, state, readout_pane):
    """Build the interactive Bokeh contour figure with click-to-sample."""
    region = REGIONS[state.region_name]
    lon0, lon1, lat0, lat1 = region
    crop = dm.crop(region)

    img_data = array_to_rgba(
        crop["hmap_m"], cmap_name=state.cmap_name,
        vmin=ELEV_MIN_M, vmax=ELEV_MAX_M,
    )
    img_data = apply_overlay(
        img_data, crop,
        show_psr=state.show_psr,
        show_slope=state.show_slope,
        show_safety=state.show_safety,
        psr_threshold=state.psr_threshold,
        show_solar_mask=state.show_solar_mask,
        solar_mask_pct=state.solar_mask_pct,
        show_golden_zone=state.show_golden_zone,
    )
    img_source = ColumnDataSource(dict(
        image=[img_data], x=[lon0], y=[lat0],
        dw=[lon1 - lon0], dh=[lat1 - lat0],
    ))

    c_levels = list(range(
        int(np.floor(ELEV_MIN_M / state.contour_interval) * state.contour_interval),
        int(ELEV_MAX_M) + state.contour_interval,
        state.contour_interval,
    ))
    xs_reg, ys_reg = extract_contour_lines(crop["hmap_m"], region, c_levels)

    a_levels = list(range(
        int(np.floor(ELEV_MIN_M / state.annot_interval) * state.annot_interval),
        int(ELEV_MAX_M) + state.annot_interval,
        state.annot_interval,
    ))
    xs_ann, ys_ann, lbl_x, lbl_y, lbl_txt = extract_contour_lines(
        crop["hmap_m"], region, a_levels, extract_labels=True
    )

    reg_source   = ColumnDataSource(dict(xs=xs_reg, ys=ys_reg))
    ann_source   = ColumnDataSource(dict(xs=xs_ann, ys=ys_ann))
    label_source = ColumnDataSource(dict(x=lbl_x, y=lbl_y, text=lbl_txt))

    if state.show_threshold:
        xs_thr, ys_thr = extract_contour_lines(
            crop["hmap_m"], region, [state.threshold_elev_m]
        )
    else:
        xs_thr, ys_thr = [], []
    thr_source = ColumnDataSource(dict(xs=xs_thr, ys=ys_thr))

    if state.show_illum_contour:
        _illum_level = state.illum_contour_pct / 100.0 * 255.0
        xs_illum, ys_illum = extract_contour_lines(
            crop["illum_lum"], region, [_illum_level]
        )
    else:
        xs_illum, ys_illum = [], []
    illum_contour_source = ColumnDataSource(dict(xs=xs_illum, ys=ys_illum))

    _fig_h = _physical_figure_height(region, target_width_px=700)

    p = figure(
        width=700,
        height=_fig_h,
        x_range=(lon0, lon1),
        y_range=(lat0, lat1),
        x_axis_label="Longitude (degrees E)",
        y_axis_label="Latitude (degrees)",
        title="Isoline / Contour Map  [click to sample  |  scroll to zoom  |  drag to pan]",
        tools="wheel_zoom,pan,reset,save",
        active_scroll="wheel_zoom",
        match_aspect=True,
        aspect_scale=1.0,
        min_border_left=40,
        background_fill_color="#1a0b3d",
        border_fill_color="#0a0a14",
    )
    p.title.text_color            = "#e0e0e0"
    p.axis.axis_label_text_color  = "#b0b0b0"
    p.axis.major_label_text_color = "#b0b0b0"

    p.image_rgba(source=img_source, image="image",
                 x="x", y="y", dw="dw", dh="dh")

    _mapper = LinearColorMapper(
        palette=_bokeh_palette(state.cmap_name),
        low=ELEV_MIN_M,
        high=ELEV_MAX_M,
    )
    _cbar = ColorBar(
        color_mapper=_mapper,
        ticker=FixedTicker(ticks=[-8000, -6000, -4000, -2000, 0, 2000, 4000, 6000]),
        label_standoff=8,
        border_line_color=None,
        location=(0, 0),
        title="Elevation (m)",
        title_text_color="#b0b0b0",
        title_text_font_size="11px",
        major_label_text_color="#b0b0b0",
        major_label_text_font_size="10px",
        background_fill_color="#0a0a14",
        bar_line_color=None,
        width=12,
        height=300,
    )
    p.add_layout(_cbar, "right")

    p.multi_line(xs="xs", ys="ys", source=reg_source,
                 line_color="white", line_width=0.4,
                 line_alpha=0.55, line_dash="dashed")

    p.multi_line(xs="xs", ys="ys", source=ann_source,
                 line_color="white", line_width=1.2, line_alpha=0.9)

    contour_labels = LabelSet(
        x="x", y="y", text="text",
        source=label_source,
        text_color="white",
        text_font_size="9px",
        background_fill_color="black",
        background_fill_alpha=0.55,
        x_offset=-14,
        y_offset=-5,
    )
    p.add_layout(contour_labels)

    p.multi_line(
        xs="xs", ys="ys", source=thr_source,
        line_color="#ef4444",
        line_width=2.5,
        line_alpha=0.95,
    )

    p.multi_line(
        xs="xs", ys="ys", source=illum_contour_source,
        line_color="#06b6d4",
        line_width=2.0,
        line_alpha=0.90,
        line_dash="solid",
    )

    p.add_tools(TapTool())
    p.add_tools(CrosshairTool(line_color="#ffffff", line_alpha=0.6))

    def on_tap(event):
        lon, lat = event.x, event.y
        res = dm.sample_point(lon, lat)
        psr_tag = "  **[PSR -- possible water ice]**" if res["is_psr"] else ""
        safety_colour = (
            "#34d399" if res["safety_score"] > 0.80 else
            "#fbbf24" if res["safety_score"] > 0.55 else
            "#f87171"
        )
        readout_pane.object = (
            f"**[Contour view]** "
            f"**Lon:** {lon:.2f} deg  **Lat:** {lat:.2f} deg{psr_tag}\n\n"
            f"**Elevation:** {res['elevation_m']:.0f} m  &nbsp;|&nbsp;  "
            f"**Illumination:** {res['illumination']:.1f}%  &nbsp;|&nbsp;  "
            f"**Slope:** {res['slope_deg']:.1f} deg  &nbsp;|&nbsp;  "
            f"**Safety:** <span style='color:{safety_colour};font-weight:bold'>"
            f"{res['safety_score']:.2f}</span>"
        )

    p.on_event(Tap, on_tap)

    def update_contour(region, crop, state):
        lon0, lon1, lat0, lat1 = region

        img_data = array_to_rgba(
            crop["hmap_m"], cmap_name=state.cmap_name,
            vmin=ELEV_MIN_M, vmax=ELEV_MAX_M,
        )
        img = apply_overlay(
            img_data, crop,
            show_psr=state.show_psr,
            show_slope=state.show_slope,
            show_safety=state.show_safety,
            psr_threshold=state.psr_threshold,
            show_solar_mask=state.show_solar_mask,
            solar_mask_pct=state.solar_mask_pct,
            show_golden_zone=state.show_golden_zone,
        )
        img_source.data = dict(
            image=[img], x=[lon0], y=[lat0],
            dw=[lon1 - lon0], dh=[lat1 - lat0],
        )

        c_levels = list(range(
            int(np.floor(ELEV_MIN_M / state.contour_interval) * state.contour_interval),
            int(ELEV_MAX_M) + state.contour_interval,
            state.contour_interval,
        ))
        xs_r, ys_r = extract_contour_lines(crop["hmap_m"], region, c_levels)
        reg_source.data = dict(xs=xs_r, ys=ys_r)

        a_levels = list(range(
            int(np.floor(ELEV_MIN_M / state.annot_interval) * state.annot_interval),
            int(ELEV_MAX_M) + state.annot_interval,
            state.annot_interval,
        ))
        xs_a, ys_a, lbl_x_r, lbl_y_r, lbl_txt_r = extract_contour_lines(
            crop["hmap_m"], region, a_levels, extract_labels=True
        )
        ann_source.data   = dict(xs=xs_a, ys=ys_a)
        label_source.data = dict(x=lbl_x_r, y=lbl_y_r, text=lbl_txt_r)

        if state.show_threshold:
            xs_t, ys_t = extract_contour_lines(
                crop["hmap_m"], region, [state.threshold_elev_m]
            )
        else:
            xs_t, ys_t = [], []
        thr_source.data = dict(xs=xs_t, ys=ys_t)

        if state.show_illum_contour:
            _lv = state.illum_contour_pct / 100.0 * 255.0
            xs_il, ys_il = extract_contour_lines(
                crop["illum_lum"], region, [_lv]
            )
        else:
            xs_il, ys_il = [], []
        illum_contour_source.data = dict(xs=xs_il, ys=ys_il)

        p.x_range.start, p.x_range.end = lon0, lon1
        p.y_range.start, p.y_range.end = lat0, lat1
        p.height = _physical_figure_height(region, target_width_px=700)
        p.width  = 700

    return p, update_contour


def build_disc_contour_map(dm, state, readout_pane):
    """
    Build the polar disc contour figure using azimuthal equidistant projection centred at the south pole.

    Always uses POLAR_DISC_REGION (-90° to -80°) regardless of the selected sub-region.
    """
    disc_region = POLAR_DISC_REGION
    region_for_disc = disc_region
    crop = dm.crop(region_for_disc)

    _R_MAX   = 10.0
    _LAT_MIN = -90.0
    _LAT_MAX = -80.0
    _DISC_PX = 600

    def _make_disc_image(crop_data, cmap_name, s_psr, s_slope, s_safety,
                         psr_thr, s_solar, solar_pct, s_golden):
        """Build equirectangular RGBA then reproject into disc pixel grid.

        The overlay crop is row-flipped to match the flip_y=True orientation of the base image.
        """
        eq_img = array_to_rgba(
            crop_data["hmap_m"], cmap_name=cmap_name,
            vmin=ELEV_MIN_M, vmax=ELEV_MAX_M,
        )

        # Row-flip crop arrays so their orientation matches eq_img (row 0 = south after flip_y=True)
        _flipped_crop = {
            "hmap_m":    np.flipud(crop_data["hmap_m"]),
            "illum_lum": np.flipud(crop_data["illum_lum"]),
            "slope_deg": np.flipud(crop_data["slope_deg"]),
            "psr_mask":  np.flipud(crop_data["psr_mask"].astype(np.uint8)).astype(bool),
            "safety":    np.flipud(crop_data["safety"]),
        }

        eq_img = apply_overlay(
            eq_img, _flipped_crop,
            show_psr=s_psr,
            show_slope=s_slope,
            show_safety=s_safety,
            psr_threshold=psr_thr,
            show_solar_mask=s_solar,
            solar_mask_pct=solar_pct,
            show_golden_zone=s_golden,
        )
        return _reproject_image_to_disc(
            eq_img, region_for_disc,
            disc_px=_DISC_PX, r_max=_R_MAX,
            lat_min=_LAT_MIN, lat_max=_LAT_MAX,
        )

    disc_img = _make_disc_image(
        crop, state.cmap_name,
        state.show_psr, state.show_slope, state.show_safety,
        state.psr_threshold,
        state.show_solar_mask, state.solar_mask_pct, state.show_golden_zone,
    )

    # Bokeh image_rgba places the image at (x, y) bottom-left with width dw, height dh in plot units
    disc_source = ColumnDataSource(dict(
        image=[disc_img],
        x=[-_R_MAX], y=[-_R_MAX],
        dw=[2 * _R_MAX], dh=[2 * _R_MAX],
    ))

    def _contour_data(crop_data, s_state):
        c_levels = list(range(
            int(np.floor(ELEV_MIN_M / s_state.contour_interval)
                * s_state.contour_interval),
            int(ELEV_MAX_M) + s_state.contour_interval,
            s_state.contour_interval,
        ))
        xs_r, ys_r = extract_contour_lines(
            crop_data["hmap_m"], region_for_disc, c_levels
        )
        xs_r, ys_r = _transform_contour_lines(
            xs_r, ys_r, r_max=_R_MAX,
            lat_min=_LAT_MIN, lat_max=_LAT_MAX,
        )

        a_levels = list(range(
            int(np.floor(ELEV_MIN_M / s_state.annot_interval)
                * s_state.annot_interval),
            int(ELEV_MAX_M) + s_state.annot_interval,
            s_state.annot_interval,
        ))
        xs_a, ys_a, lbl_lon, lbl_lat, lbl_txt = extract_contour_lines(
            crop_data["hmap_m"], region_for_disc, a_levels, extract_labels=True
        )
        xs_a, ys_a = _transform_contour_lines(
            xs_a, ys_a, r_max=_R_MAX,
            lat_min=_LAT_MIN, lat_max=_LAT_MAX,
        )
        lbl_x_d, lbl_y_d = _lonlat_to_disc(
            lbl_lon, lbl_lat, r_max=_R_MAX,
            lat_min=_LAT_MIN, lat_max=_LAT_MAX,
        )
        return xs_r, ys_r, xs_a, ys_a, lbl_x_d.tolist(), lbl_y_d.tolist(), lbl_txt

    xs_r, ys_r, xs_a, ys_a, lbl_x, lbl_y, lbl_txt = _contour_data(crop, state)

    reg_source   = ColumnDataSource(dict(xs=xs_r, ys=ys_r))
    ann_source   = ColumnDataSource(dict(xs=xs_a, ys=ys_a))
    label_source = ColumnDataSource(dict(x=lbl_x, y=lbl_y, text=lbl_txt))

    if state.show_threshold:
        xs_t, ys_t = extract_contour_lines(
            crop["hmap_m"], region_for_disc, [state.threshold_elev_m]
        )
        xs_t, ys_t = _transform_contour_lines(
            xs_t, ys_t, r_max=_R_MAX, lat_min=_LAT_MIN, lat_max=_LAT_MAX
        )
    else:
        xs_t, ys_t = [], []
    thr_source = ColumnDataSource(dict(xs=xs_t, ys=ys_t))

    if state.show_illum_contour:
        _lv = state.illum_contour_pct / 100.0 * 255.0
        xs_il, ys_il = extract_contour_lines(
            crop["illum_lum"], region_for_disc, [_lv]
        )
        xs_il, ys_il = _transform_contour_lines(
            xs_il, ys_il, r_max=_R_MAX, lat_min=_LAT_MIN, lat_max=_LAT_MAX
        )
    else:
        xs_il, ys_il = [], []
    illum_src = ColumnDataSource(dict(xs=xs_il, ys=ys_il))

    # Square figure: disc fits in a square with side 2*R_MAX plus margin
    _margin = 0.5
    _ax_range = (-_R_MAX - _margin, _R_MAX + _margin)

    p = figure(
        width=660, height=660,
        x_range=_ax_range,
        y_range=_ax_range,
        title=(
            "Polar Disc — Isoline Map  (azimuthal equidistant, -90° to -80°)  "
            "[click to sample  |  scroll to zoom  |  drag to pan]"
        ),
        tools="wheel_zoom,pan,reset,save",
        active_scroll="wheel_zoom",
        match_aspect=True,
        aspect_scale=1.0,
        background_fill_color="#0a0a14",
        border_fill_color="#0a0a14",
    )
    p.title.text_color             = "#e0e0e0"
    p.title.text_font_size         = "12px"
    # Suppress raw plot-unit tick labels; coordinates shown via click-to-sample readout instead
    p.xaxis.major_label_text_font_size = "0pt"
    p.yaxis.major_label_text_font_size = "0pt"
    p.xaxis.axis_label = ""
    p.yaxis.axis_label = ""

    import math as _math
    _theta_pts = np.linspace(0, 2 * _math.pi, 200)
    p.line(
        x=(_R_MAX * np.sin(_theta_pts)).tolist(),
        y=(_R_MAX * np.cos(_theta_pts)).tolist(),
        line_color="#475569", line_width=1.0, line_alpha=0.7,
    )
    p.scatter(x=[0], y=[0], marker="cross", size=8,
              line_color="#94a3b8", line_width=1.5)

    p.image_rgba(source=disc_source, image="image",
                 x="x", y="y", dw="dw", dh="dh")

    p.multi_line(xs="xs", ys="ys", source=reg_source,
                 line_color="white", line_width=0.4,
                 line_alpha=0.55, line_dash="dashed")
    p.multi_line(xs="xs", ys="ys", source=ann_source,
                 line_color="white", line_width=1.2, line_alpha=0.9)
    p.add_layout(LabelSet(
        x="x", y="y", text="text", source=label_source,
        text_color="black", text_font_size="11px",
        text_font_style="bold",
        x_offset=-14, y_offset=-5,
    ))
    p.multi_line(xs="xs", ys="ys", source=thr_source,
                 line_color="#ef4444", line_width=2.5, line_alpha=0.95)
    p.multi_line(xs="xs", ys="ys", source=illum_src,
                 line_color="#06b6d4", line_width=2.0, line_alpha=0.90)

    _site_lons  = [v[0] for v in ARTEMIS_SITE_CENTRES.values()]
    _site_lats  = [v[1] for v in ARTEMIS_SITE_CENTRES.values()]
    _site_names = list(ARTEMIS_SITE_CENTRES.keys())
    _site_xs, _site_ys = _lonlat_to_disc(
        _site_lons, _site_lats,
        r_max=_R_MAX, lat_min=_LAT_MIN, lat_max=_LAT_MAX,
    )
    _valid = [i for i, lat in enumerate(_site_lats)
              if _LAT_MIN <= lat <= _LAT_MAX]
    _site_xs_v  = _site_xs[_valid].tolist()
    _site_ys_v  = _site_ys[_valid].tolist()
    _site_nm_v  = [_site_names[i] for i in _valid]

    site_disc_source = ColumnDataSource(dict(
        x=_site_xs_v, y=_site_ys_v, name=_site_nm_v,
    ))

    _site_scatter = p.scatter(
        x="x", y="y", source=site_disc_source,
        marker="star", size=11,
        fill_color="#facc15", line_color="#ffffff", line_width=0.8,
        fill_alpha=0.95,
        visible=state.show_disc_sites,
    )

    _site_short = [n if len(n) <= 16 else n[:14] + "\u2026" for n in _site_nm_v]
    site_disc_source.data["short"] = _site_short

    _site_labels = LabelSet(
        x="x", y="y", text="short", source=site_disc_source,
        text_color="black", text_font_size="10px",
        text_font_style="bold",
        x_offset=5, y_offset=4,
        visible=state.show_disc_sites,
    )
    p.add_layout(_site_labels)

    # Coordinate display label at top of disc; populated by the CustomJS mousemove callback
    coord_ds = ColumnDataSource(dict(
        x=[0.0], y=[_R_MAX * 0.82], text=[""],
    ))
    p.add_layout(LabelSet(
        x="x", y="y", text="text", source=coord_ds,
        text_align="center",
        text_color="#facc15",
        text_font_size="12px",
        text_font_style="bold",
        background_fill_color="#0f172a",
        background_fill_alpha=0.80,
        border_line_color="#334155",
        border_line_width=1,
    ))

    # Debounced 2-second mousemove callback: inverse azimuthal equidistant transform in JS
    _coord_js = CustomJS(
        args=dict(ds=coord_ds, r_max=_R_MAX,
                  lat_min=_LAT_MIN, lat_max=_LAT_MAX),
        code="""
        clearTimeout(window._disc_coord_timer);
        const x = cb_obj.x;
        const y = cb_obj.y;
        ds.data['text'] = [''];
        ds.change.emit();
        window._disc_coord_timer = setTimeout(function() {
            const r = Math.sqrt(x * x + y * y);
            if (r > r_max) { return; }
            const lat = lat_min + (r / r_max) * (lat_max - lat_min);
            const theta = Math.atan2(x, y);
            const lon = ((theta * 180.0 / Math.PI) % 360.0 + 360.0) % 360.0;
            ds.data['text'] = [
                lon.toFixed(1) + '\u00b0E  \u00a0  ' + lat.toFixed(1) + '\u00b0'
            ];
            ds.change.emit();
        }, 2000);
        """
    )
    p.js_on_event("mousemove", _coord_js)

    p.add_tools(TapTool())
    p.add_tools(CrosshairTool(line_color="#ffffff", line_alpha=0.6))

    def on_tap(event):
        result = _disc_to_lonlat(event.x, event.y,
                                  r_max=_R_MAX,
                                  lat_min=_LAT_MIN, lat_max=_LAT_MAX)
        if result is None:
            return
        lon, lat = result
        res = dm.sample_point(lon, lat)
        psr_tag = "  **[PSR -- possible water ice]**" if res["is_psr"] else ""
        safety_colour = (
            "#34d399" if res["safety_score"] > 0.80 else
            "#fbbf24" if res["safety_score"] > 0.55 else
            "#f87171"
        )
        readout_pane.object = (
            f"**[Disc view]** "
            f"**Lon:** {lon:.2f} deg  **Lat:** {lat:.2f} deg{psr_tag}\n\n"
            f"**Elevation:** {res['elevation_m']:.0f} m  &nbsp;|&nbsp;  "
            f"**Illumination:** {res['illumination']:.1f}%  &nbsp;|&nbsp;  "
            f"**Slope:** {res['slope_deg']:.1f} deg  &nbsp;|&nbsp;  "
            f"**Safety:** <span style='color:{safety_colour};"
            f"font-weight:bold'>{res['safety_score']:.2f}</span>"
        )

    p.on_event(Tap, on_tap)

    def update_disc(region_unused, crop_unused, s):
        # Disc always re-crops POLAR_DISC_REGION regardless of the selected sub-region
        _crop = dm.crop(region_for_disc)

        new_img = _make_disc_image(
            _crop, s.cmap_name,
            s.show_psr, s.show_slope, s.show_safety,
            s.psr_threshold,
            s.show_solar_mask, s.solar_mask_pct, s.show_golden_zone,
        )
        disc_source.data = dict(
            image=[new_img],
            x=[-_R_MAX], y=[-_R_MAX],
            dw=[2 * _R_MAX], dh=[2 * _R_MAX],
        )

        xs_r2, ys_r2, xs_a2, ys_a2, lx2, ly2, lt2 = _contour_data(_crop, s)
        reg_source.data   = dict(xs=xs_r2, ys=ys_r2)
        ann_source.data   = dict(xs=xs_a2, ys=ys_a2)
        label_source.data = dict(x=lx2, y=ly2, text=lt2)

        if s.show_threshold:
            xs_t2, ys_t2 = extract_contour_lines(
                _crop["hmap_m"], region_for_disc, [s.threshold_elev_m]
            )
            xs_t2, ys_t2 = _transform_contour_lines(
                xs_t2, ys_t2, r_max=_R_MAX,
                lat_min=_LAT_MIN, lat_max=_LAT_MAX,
            )
        else:
            xs_t2, ys_t2 = [], []
        thr_source.data = dict(xs=xs_t2, ys=ys_t2)

        if s.show_illum_contour:
            _lv2 = s.illum_contour_pct / 100.0 * 255.0
            xs_il2, ys_il2 = extract_contour_lines(
                _crop["illum_lum"], region_for_disc, [_lv2]
            )
            xs_il2, ys_il2 = _transform_contour_lines(
                xs_il2, ys_il2, r_max=_R_MAX,
                lat_min=_LAT_MIN, lat_max=_LAT_MAX,
            )
        else:
            xs_il2, ys_il2 = [], []
        illum_src.data = dict(xs=xs_il2, ys=ys_il2)

    def set_sites_visible(visible: bool):
        """Show or hide the Artemis site markers on the disc."""
        _site_scatter.visible = visible
        _site_labels.visible  = visible

    return p, update_disc, set_sites_visible


def create_app():
    """Factory function called once per browser session by pn.serve()."""
    state = AppState()

    readout_pane = pn.pane.Markdown(
        "_Click the map to sample elevation, illumination, slope, and safety score._",
        styles={
            "color": "#d0d8e8",
            "font-family": "monospace",
            "font-size": "13px",
            "background": "#111827",
            "padding": "10px",
            "border-radius": "6px",
            "min-height": "60px",
        },
        width=700,
    )

    plotly_3d_pane = pn.pane.Plotly(
        go.Figure(layout=dict(
            paper_bgcolor="#0f172a",
            height=540,
            annotations=[dict(
                text="Press  Render Maps  to load the 3D interactive terrain",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="#475569", size=13),
            )],
        )),
        config={"scrollZoom": True, "displayModeBar": True},
        height=540,
        sizing_mode="stretch_width",
    )

    status_pane = pn.pane.Markdown(
        "**Status:** Ready — press **Render** to generate maps",
        styles={"color": "#86efac", "font-size": "12px"},
        width=240,
    )

    _last_click = {"lon": None, "lat": None, "score": None}
    bokeh_map, update_bokeh, pin_source = build_bokeh_map(DM, state, readout_pane, _last_click)
    bokeh_pane = pn.pane.Bokeh(bokeh_map)

    contour_map, update_contour = build_contour_map(DM, state, readout_pane)
    contour_pane = pn.pane.Bokeh(contour_map)

    disc_map, update_disc, set_disc_sites_visible = \
        build_disc_contour_map(DM, state, readout_pane)
    disc_map_pane = pn.pane.Bokeh(disc_map)

    _render_lock = threading.Lock()

    def trigger_render():
        if not _render_lock.acquire(blocking=False):
            status_pane.object = "**Status:** Already rendering, please wait..."
            return
        status_pane.object = "**Status:** Rendering..."
        region = REGIONS[state.region_name]
        crop   = DM.crop(region)

        update_bokeh(region, crop, state)
        update_contour(region, crop, state)

        def worker():
            try:
                fig_3d = RenderEngine.render_3d_plotly(
                    crop, region,
                    vert_exag=state.plotly_vert_exag,
                    cmap_name=state.cmap_name,
                    show_illumination_drape=state.show_illum_drape,
                    show_threshold=state.show_threshold,
                    threshold_elev_m=state.threshold_elev_m,
                )
                plotly_3d_pane.object = fig_3d
                status_pane.object = "**Status:** Done"
            except Exception as exc:
                status_pane.object = f"**Status:** Error — {exc}"
            finally:
                _render_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    render_btn = pn.widgets.Button(
        name="▶ Render Maps", button_type="primary", width=220
    )
    render_btn.on_click(lambda _: trigger_render())

    mark_btn = pn.widgets.Button(
        name="Mark as candidate site",
        button_type="success",
        width=220,
    )

    def on_mark_candidate(_):
        lon   = _last_click.get("lon")
        lat   = _last_click.get("lat")
        score = _last_click.get("score", 0.0)
        if lon is None:
            status_pane.object = "**Status:** Click the map first to select a location."
            return
        n      = len(_CANDIDATE_PINS) + 1
        label  = f"C{n}"
        colour = "#34d399" if score > 0.80 else "#fbbf24"
        _CANDIDATE_PINS.append({"lon": lon, "lat": lat, "label": label, "score": score})
        pin_source.stream(dict(x=[lon], y=[lat], label=[label], color=[colour]))
        status_pane.object = (
            f"**Status:** Candidate {label} pinned at "
            f"({lon:.2f} deg, {lat:.2f} deg)  safety={score:.2f}"
        )

    mark_btn.on_click(on_mark_candidate)

    sync_3d_btn = pn.widgets.Button(
        name="Sync 3D to map view",
        button_type="warning",
        width=220,
    )

    def on_sync_3d(_):
        """Read the current Bokeh map extent and use it as the region for a fresh Plotly 3D render."""
        lon0 = max(float(bokeh_map.x_range.start),   0.0)
        lon1 = min(float(bokeh_map.x_range.end),   360.0)
        lat0 = max(float(bokeh_map.y_range.start), -89.9)
        lat1 = min(float(bokeh_map.y_range.end),   -60.0)

        if lon1 - lon0 < 0.5 or lat1 - lat0 < 0.2:
            status_pane.object = (
                "**Status:** Map view too small to sync — zoom out slightly."
            )
            return

        live_region = [lon0, lon1, lat0, lat1]

        if not _render_lock.acquire(blocking=False):
            status_pane.object = "**Status:** Already rendering, please wait..."
            return

        status_pane.object = "**Status:** Syncing 3D to map view..."

        def _worker():
            try:
                crop   = DM.crop(live_region)
                fig_3d = RenderEngine.render_3d_plotly(
                    crop, live_region,
                    vert_exag=state.plotly_vert_exag,
                    cmap_name=state.cmap_name,
                    show_illumination_drape=state.show_illum_drape,
                    show_threshold=state.show_threshold,
                    threshold_elev_m=state.threshold_elev_m,
                )
                plotly_3d_pane.object = fig_3d
                status_pane.object = (
                    f"**Status:** 3D synced to "
                    f"lon {lon0:.1f}–{lon1:.1f} deg, "
                    f"lat {lat0:.1f}–{lat1:.1f} deg"
                )
            except Exception as exc:
                status_pane.object = f"**Status:** Sync error — {exc}"
            finally:
                _render_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    sync_3d_btn.on_click(on_sync_3d)

    def _on_3d_surface_click(event):
        """Handle Plotly click events on the 3D surface; x/y data coordinates map directly to lon/lat."""
        click_data = event.new
        if not click_data:
            return
        try:
            pt  = click_data["points"][0]
            lon = float(pt["x"])
            lat = float(pt["y"])
            res = DM.sample_point(lon, lat)
            psr_tag = "  **[PSR -- possible water ice]**" if res["is_psr"] else ""
            safety_colour = (
                "#34d399" if res["safety_score"] > 0.80 else
                "#fbbf24" if res["safety_score"] > 0.55 else
                "#f87171"
            )
            readout_pane.object = (
                f"**[3D surface]** "
                f"**Lon:** {lon:.2f} deg  **Lat:** {lat:.2f} deg{psr_tag}\n\n"
                f"**Elevation:** {res['elevation_m']:.0f} m  &nbsp;|&nbsp;  "
                f"**Illumination:** {res['illumination']:.1f}%  &nbsp;|&nbsp;  "
                f"**Slope:** {res['slope_deg']:.1f} deg  &nbsp;|&nbsp;  "
                f"**Safety:** <span style='color:{safety_colour};"
                f"font-weight:bold'>{res['safety_score']:.2f}</span>"
            )
            _last_click["lon"]   = lon
            _last_click["lat"]   = lat
            _last_click["score"] = res["safety_score"]
        except (KeyError, IndexError, TypeError):
            pass

    plotly_3d_pane.param.watch(_on_3d_surface_click, "click_data")

    region_sel = pn.widgets.Select(
        name="Region preset", options=list(REGIONS.keys()),
        value=state.region_name, width=220,
    )
    cint_sl = pn.widgets.IntSlider(
        name="Contour interval (m)", start=100, end=2000, step=100,
        value=state.contour_interval, width=220,
    )
    aint_sl = pn.widgets.IntSlider(
        name="Annotation every (m)", start=500, end=5000, step=500,
        value=state.annot_interval, width=220,
    )
    def _make_toggle(base_name, initial_value, button_type, width=220):
        """Create a Toggle widget whose label shows ON/OFF state."""
        suffix = "  ▶ ON" if initial_value else "  ◼ OFF"
        tog = pn.widgets.Toggle(
            name=base_name + suffix,
            value=initial_value,
            button_type=button_type,
            width=width,
        )
        def _update_label(event):
            tog.name = base_name + ("  ▶ ON" if event.new else "  ◼ OFF")
        tog.param.watch(_update_label, "value")
        return tog

    psr_tog    = _make_toggle("PSR overlay",    state.show_psr,    "primary", width=105)
    slope_tog  = _make_toggle("Slope overlay",  state.show_slope,  "warning", width=105)
    safety_tog = _make_toggle("Safety overlay", state.show_safety, "success", width=220)
    psr_thr_sl = pn.widgets.IntSlider(
        name="PSR threshold (luminance 0-255)",
        start=10, end=120, step=5, value=state.psr_threshold, width=220,
    )
    cmap_sel = pn.widgets.Select(
        name="Colourmap", options=CMAP_OPTIONS,
        value=state.cmap_name, width=220,
    )

    illum_drape_tog = _make_toggle(
        "Drape illumination", state.show_illum_drape, "primary", width=220
    )
    plotly_vexag_sl = pn.widgets.FloatSlider(
        name="3D vertical exaggeration",
        start=0.5, end=10.0, step=0.5,
        value=state.plotly_vert_exag,
        width=220,
    )
    thr_tog = _make_toggle("Threshold contour", state.show_threshold, "danger", width=220)
    thr_sl = pn.widgets.IntSlider(
        name="Threshold elevation (m)",
        start=-9000, end=7000, step=100,
        value=state.threshold_elev_m,
        width=220,
    )

    illum_contour_tog = _make_toggle(
        "Solar contour line", state.show_illum_contour, "primary", width=220
    )
    illum_contour_sl = pn.widgets.IntSlider(
        name="Solar contour level (%)",
        start=10, end=100, step=5,
        value=state.illum_contour_pct,
        width=220,
    )
    solar_mask_tog = _make_toggle(
        "Solar suitability mask", state.show_solar_mask, "warning", width=220
    )
    solar_mask_sl = pn.widgets.IntSlider(
        name="Min solar exposure (%)",
        start=10, end=100, step=5,
        value=state.solar_mask_pct,
        width=220,
    )
    golden_zone_tog = _make_toggle(
        "Golden zone (sunlit + flat)", state.show_golden_zone, "success", width=220
    )

    def _sync(widget, attr, key):
        def cb(e):
            setattr(state, key, e.new)
        widget.param.watch(cb, attr)

    _sync(region_sel, "value", "region_name")
    _sync(cint_sl,    "value", "contour_interval")
    _sync(aint_sl,    "value", "annot_interval")
    _sync(psr_tog,    "value", "show_psr")
    _sync(slope_tog,  "value", "show_slope")
    _sync(safety_tog, "value", "show_safety")
    _sync(psr_thr_sl,      "value", "psr_threshold")
    _sync(cmap_sel,        "value", "cmap_name")
    _sync(illum_drape_tog, "value", "show_illum_drape")
    _sync(plotly_vexag_sl, "value", "plotly_vert_exag")
    _sync(thr_tog, "value", "show_threshold")
    _sync(thr_sl,  "value", "threshold_elev_m")
    _sync(illum_contour_tog, "value", "show_illum_contour")
    _sync(illum_contour_sl,  "value", "illum_contour_pct")
    _sync(solar_mask_tog,    "value", "show_solar_mask")
    _sync(solar_mask_sl,     "value", "solar_mask_pct")
    _sync(golden_zone_tog,   "value", "show_golden_zone")

    disc_sites_tog = _make_toggle(
        "Artemis sites on disc", state.show_disc_sites, "primary", width=220
    )
    _sync(disc_sites_tog, "value", "show_disc_sites")

    def _on_disc_sites_change(event):
        set_disc_sites_visible(event.new)

    disc_sites_tog.param.watch(_on_disc_sites_change, "value")

    def _on_contour_param_change(_):
        region = REGIONS[state.region_name]
        crop   = DM.crop(region)
        update_bokeh(region, crop, state)
        update_contour(region, crop, state)
        update_disc(region, crop, state)

    for _param_name in [
        "show_psr", "psr_threshold",
        "show_slope", "show_safety",
        "contour_interval", "annot_interval",
        "show_threshold", "threshold_elev_m",
        "show_illum_contour", "illum_contour_pct",
        "show_solar_mask", "solar_mask_pct",
        "show_golden_zone",
    ]:
        state.param.watch(_on_contour_param_change, _param_name)

    def _on_plotly_param_change(_):
        trigger_render()

    for _param_name in ["show_illum_drape", "plotly_vert_exag"]:
        state.param.watch(_on_plotly_param_change, _param_name)

    def _section(label, *widgets):
        """Render a titled group of controls."""
        return pn.Column(
            pn.pane.Markdown(f"### {label}",
                             styles={"color": "#93c5fd", "margin": "6px 0 2px 0"}),
            *widgets,
        )

    def _subsection(label, *widgets):
        """Render a lighter sub-heading for per-visualisation groups."""
        return pn.Column(
            pn.pane.Markdown(
                f"**{label}**",
                styles={"color": "#7dd3fc", "font-size": "12px",
                        "margin": "8px 0 2px 0"},
            ),
            *widgets,
        )

    sidebar = pn.Column(
        _section("Region", region_sel),
        pn.layout.Divider(),
        _section("Colourmap", cmap_sel),
        pn.layout.Divider(),
        render_btn,
        status_pane,
        pn.layout.Divider(),

        _section(
            "Interactive Map",
            _subsection(
                "Overlays",
                pn.Row(psr_tog, slope_tog),
                safety_tog,
                psr_thr_sl,
            ),
            _subsection(
                "Solar analysis",
                solar_mask_tog,
                solar_mask_sl,
                golden_zone_tog,
            ),
        ),
        pn.layout.Divider(),
        _section(
            "Isoline / Contour",
            cint_sl,
            aint_sl,
            _subsection(
                "Elevation threshold",
                thr_tog,
                thr_sl,
                pn.pane.Markdown(
                    "_Draws a bold red contour at this elevation on both "
                    "the 2D map and the 3D surface. Use to mark a zone "
                    "boundary (e.g. -3000 m crater floor limit)._",
                    styles={"color": "#6b7280", "font-size": "10px"},
                ),
            ),
            _subsection(
                "Illumination analysis",
                illum_contour_tog,
                illum_contour_sl,
                pn.pane.Markdown(
                    "_Cyan iso-line at this solar exposure level. "
                    "Use 80% to mark the minimum viable solar power zone._",
                    styles={"color": "#6b7280", "font-size": "10px"},
                ),
            ),
            _subsection(
                "Polar disc",
                disc_sites_tog,
                pn.pane.Markdown(
                    "_Show or hide the 13 Artemis candidate site markers "
                    "on the polar disc contour map._",
                    styles={"color": "#6b7280", "font-size": "10px"},
                ),
            ),
        ),
        pn.layout.Divider(),
        _section(
            "3D Interactive Surface",
            plotly_vexag_sl,
            illum_drape_tog,
            sync_3d_btn,
            pn.pane.Markdown(
                "_Pan/zoom the interactive map first, then press._",
                styles={"color": "#6b7280", "font-size": "10px"},
            ),
        ),
        pn.layout.Divider(),
        _section("Landing Candidates", mark_btn),
        width=250,
        styles={
            "background": "#111827",
            "padding": "12px",
            "border-radius": "8px",
        },
    )

    legend_pane = pn.pane.HTML(
        """
<div style="display:grid;grid-template-columns:1fr 1fr;gap:3px 16px;
            background:#111827;padding:8px 12px;border-radius:4px;
            font-size:11px;color:#d1d5db;line-height:1.9;">
  <span><span style="color:#3b82f6">&#9632;</span> PSR overlay (possible water ice)</span>
  <span><span style="color:#facc15">&#9632;</span> Solar suitability mask (yellow)</span>
  <span><span style="color:#f97316">&#9632;</span> Slope 10–15° (caution)</span>
  <span><span style="color:#d97706">&#9632;</span> Golden zone: sunlit + flat</span>
  <span><span style="color:#dc2626">&#9632;</span> Slope &gt;15° (unsafe)</span>
  <span><span style="color:#06b6d4">&#9632;</span> Solar contour iso-line (cyan)</span>
  <span><span style="color:#34d399">&#9632;</span> Safety score &gt;0.80 (candidate)</span>
  <span><span style="color:#ef4444">&#9632;</span> Elevation threshold contour (red)</span>
  <span><span style="color:#94a3b8">&#9632;</span> Illumination drape (toggle in sidebar)</span>
  <span><span style="color:#facc15">&#9632;</span> Artemis sites (disc, toggleable)</span>
</div>
""",
        sizing_mode="stretch_width",
    )

    main = pn.Column(
        pn.pane.Markdown(
            "## Artemis III — Lunar South Pole Landing Site Explorer",
            styles={"color": "#e2e8f0", "font-size": "20px",
                    "letter-spacing": "0.03em"},
        ),
        legend_pane,

        pn.Column(
            pn.pane.Markdown(
                "#### Interactive Map",
                styles={"color": "#d1d5db"},
            ),
            bokeh_pane,
            readout_pane,
        ),

        pn.layout.Divider(),

        pn.Column(
            pn.pane.Markdown(
                "#### 3D Interactive Terrain  _(Plotly WebGL — drag to rotate · scroll to zoom · click to sample)_",
                styles={"color": "#d1d5db"},
            ),
            plotly_3d_pane,
        ),
        pn.layout.Divider(),
        pn.Column(
            pn.pane.Markdown(
                "#### Isoline / Contour Map  _(Bokeh · equirectangular · click to sample)_",
                styles={"color": "#d1d5db"},
            ),
            contour_pane,
        ),
        pn.layout.Divider(),
        pn.Column(
            pn.pane.Markdown(
                "#### Polar Disc — Isoline Map  _(azimuthal equidistant · -90° to -80° · hover 2s for coords · click to sample)_",
                styles={"color": "#d1d5db"},
            ),
            disc_map_pane,
            readout_pane,
        ),
        styles={"background": "#0f172a", "padding": "16px"},
    )

    template = pn.template.FastListTemplate(
        title="Artemis III — Technical Terrain Explorer",
        sidebar=[sidebar],
        main=[main],
        theme="dark",
        accent="#3b82f6",
        header_background="#0f172a",
    )

    trigger_render()

    return template


if __name__ == "__main__":
    missing = [p for p in [HEIGHTMAP_PATH, ILLUMINATION_PATH]
               if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
        print(
            "\n  Ensure the dataset/ folder is present at the project root\n"
            "  (see README.md, \"Getting the data\", for the expected layout).",
            file=sys.stderr,
        )
        sys.exit(1)

    _arr = np.array(Image.open(HEIGHTMAP_PATH))
    _fname = os.path.basename(HEIGHTMAP_PATH)
    if _arr.dtype in (np.uint16, np.int16):
        _elev_min = float(_arr.min()) * 0.5 - 8000.0
        _elev_max = float(_arr.max()) * 0.5 - 8000.0
        print(
            f"[calibration] {_fname}  dtype={_arr.dtype}  "
            f"shape={_arr.shape}  "
            f"DN range=[{_arr.min()}, {_arr.max()}]  "
            f"elevation range=[{_elev_min:.0f} m, {_elev_max:.0f} m]",
            flush=True,
        )
    else:
        print(
            f"[calibration] {_fname}  dtype={_arr.dtype}  "
            f"shape={_arr.shape}  "
            f"min={_arr.min():.3f}  max={_arr.max():.3f}",
            flush=True,
        )
    del _arr

    # Remove stale GMT session dirs from crashed runs; skip the active PID's own session
    import shutil as _shutil
    _gmt_sessions = os.path.expanduser("~/.gmt/sessions")
    _current_pid = str(os.getpid())
    if os.path.isdir(_gmt_sessions):
        _removed = 0
        for _entry in os.listdir(_gmt_sessions):
            _parts = _entry.rsplit(".", 1)
            if len(_parts) == 2 and _parts[1] != _current_pid:
                try:
                    _pid_int = int(_parts[1])
                    os.kill(_pid_int, 0)
                except (ProcessLookupError, PermissionError, ValueError):
                    _shutil.rmtree(
                        os.path.join(_gmt_sessions, _entry), ignore_errors=True
                    )
                    _removed += 1
        if _removed:
            print(f"[startup] Removed {_removed} stale GMT session(s).", flush=True)

    print("[startup] Loading dataset (10–20 s on first run) ...", flush=True)
    DM = DataManager()

    print("[startup] Starting Panel server at http://localhost:5006", flush=True)
    print("[startup] Press Ctrl+C to stop.", flush=True)

    pn.serve(
        create_app,
        port=5006,
        show=True,
        title="Artemis III Landing Site Explorer",
        autoreload=False,
    )
