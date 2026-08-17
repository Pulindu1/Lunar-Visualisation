"""
narrative_infographic.py  --  Artemis III: Understanding the Lunar South Pole

A guided, story-driven dashboard for a non-expert audience, explaining why the
south pole was chosen as the Artemis III destination.

Run:
    python3 src/narrative_infographic.py

Opens a browser at http://localhost:5007 automatically.
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
    ColumnDataSource,
    LinearColorMapper, LabelSet,
    Span, Label,
)
from bokeh.events import Tap

import xarray as xr
import pygmt
from scipy.ndimage import zoom as scipy_zoom
from scipy.ndimage import gaussian_filter as scipy_gaussian
import re
import matplotlib.pyplot as plt

try:
    import cmcrameri.cm as cmc
    _HAS_CMC = True
except ImportError:
    _HAS_CMC = False

warnings.filterwarnings("ignore")   # suppress PyGMT PostScript backend noise

# Data lives in dataset/ at the project root, one level above this file, so the
# app can be launched from any working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEIGHTMAP_PATH    = os.path.join(PROJECT_ROOT, "dataset", "heightmaps",   "ldem_4.tif")
ILLUMINATION_PATH = os.path.join(PROJECT_ROOT, "dataset", "illumination", "lroc_color_poles_4k.tif")

LON_MIN, LON_MAX =   0.0, 360.0
LAT_MAX, LAT_MIN =  90.0, -90.0
ELEV_MIN_M       = -9000.0
ELEV_MAX_M       =  7000.0
SLOPE_DANGER_DEG =  15.0
GROUND_RES_M_PER_PX = 7600.0

FULL_POLE_REGION = [1, 359, -90, -60]

# GMT requires full [0, 360] span for seamless azimuthal equidistant disc.
FULL_POLE_REGION_GMT = [0, 360, -90, -60]

# Tighter crop for the illumination disc; lon=1 avoids Bokeh black-strip artefact at lon=0.
ILLUM_DISC_CROP_REGION = [1, 359, -90, -83]

ARTEMIS_SITES = {
    "Shackleton crater":         [160, -89.9,
        "On the rim of a 21-km deep crater. Almost permanent sunlight "
        "on the rim, permanent darkness inside. Possible water ice."],
    "Peak near Shackleton":      [155, -89.0,
        "An elevated ridge with some of the longest sunlight hours "
        "near the south pole. Strong candidate for a solar power base."],
    "Connecting Ridge":          [160, -88.8,
        "A natural bridge of high terrain linking two crater rims. "
        "Good sunlight and close to shadowed ice deposits."],
    "Connecting Ridge Extended": [155, -88.5,
        "A broader section of the ridge system. More flat terrain "
        "available for landing than the narrower ridge crest."],
    "de Gerlache Rim 1":         [300, -88.7,
        "Western rim of de Gerlache crater. Elevated terrain with "
        "good Earth visibility for communications."],
    "de Gerlache Rim 2":         [320, -88.7,
        "Eastern section of de Gerlache crater rim. Slightly higher "
        "elevation and better solar access than Rim 1."],
    "de Gerlache-Kocher Massif": [290, -87.0,
        "A large elevated plateau between two major craters. "
        "Relatively flat with moderate illumination."],
    "Haworth":                   [20,  -88.5,
        "Inside a cluster of small craters. Close to permanently "
        "shadowed regions that may contain water ice deposits."],
    "Malapert Massif":           [15,  -85.5,
        "A tall mountain south of Malapert crater. Proposed location "
        "for a communications relay due to line-of-sight to Earth."],
    "Leibnitz Beta Plateau":     [180, -85.5,
        "A broad, relatively flat plateau with consistent illumination. "
        "One of the flatter candidate areas near the pole."],
    "Nobile Rim 1":              [75,  -88.8,
        "Western rim of Nobile crater. The crater floor is one of the "
        "largest permanently shadowed regions and likely holds ice."],
    "Nobile Rim 2":              [100, -88.8,
        "Eastern rim of Nobile crater. Similar to Rim 1 but with "
        "slightly different illumination geometry."],
    "Amundsen / Faustini Rim":   [95,  -87.0,
        "Elevated terrain on the rim of Amundsen and Faustini craters. "
        "Good candidate for a science base near ice deposits."],
}

# Pre-attentively distinct shape icons encoding primary mission value per site.
SITE_ICONS = {
    "Shackleton crater":         "❄",
    "Peak near Shackleton":      "☀",
    "Connecting Ridge":          "☀",
    "Connecting Ridge Extended": "☀",
    "de Gerlache Rim 1":         "❄",
    "de Gerlache Rim 2":         "☀",
    "de Gerlache-Kocher Massif": "☀",
    "Haworth":                   "❄",
    "Malapert Massif":           "⬡",
    "Leibnitz Beta Plateau":     "☀",
    "Nobile Rim 1":              "❄",
    "Nobile Rim 2":              "❄",
    "Amundsen / Faustini Rim":   "❄",
}

CMAP_OPTIONS = ["batlow", "oleron", "lapaz"]

ILLUM_CMAP_OPTIONS = ["cividis", "plasma", "gray"]

TOUR_SITES = [
    (
        "Peak near Shackleton",
        "**Stop 1 of 3 — Peak near Shackleton**\n\n"
        "The elevated peak sits just very close to the south pole and is one of the "
        "most sunlit points on the entire Moon, receiving sunlight for up to "
        "94% of the lunar year. And there is a crater floor below it. "
        "Its high elevation and near-polar "
        "location make it an ideal anchor for a solar power station that could "
        "supply continuous energy to a crewed base, while Shackleton's ice-filled "
        "crater floor lies within rover range just kilometres away.",
    ),
    (
        "Connecting Ridge",
        "**Stop 2 of 3 — Connecting Ridge**\n\n"
        "This narrow ridge links the rims of Shackleton and de Gerlache craters. "
        "Its elevation keeps it above the low-angle shadows that swallow crater "
        "floors. The ridge receives consistent sunlight and offers relatively "
        "flat terrain, with much flatter terrain a little further out for landing. "
        "It is close enough to both craters' shadowed floors to allow rover "
        "excursions to retrieve ice samples without leaving the sunlit zone for extended periods of time.",
    ),
    (
        "Malapert Massif",
        "**Stop 3 of 3 —  Malapert Massif**\n\n"
        "Malapert is a mountain, not a crater rim. Its height of approximately "
        "5 km gives it an unobstructed line of sight to Earth, whislts getting a large amount of sunlight. "
        "It is also one of the flatter high-"
        "altitude sites, making it a realistic landing target. Nearby are also several craters which could be explored.",
    ),
]

SITE_CARD_PARAMS = {
    "Peak near Shackleton": {
        "region":        [100, 220, -90.0, -88.5],
        "azimuth":       160,
        "elev_angle":    40,
        "zsize":         "4c",
        "use_psr_drape": True,
    },
    "Connecting Ridge": {
        "region":        [110, 210, -90.0, -87.0],
        "azimuth":       130,
        "elev_angle":    30,
        "zsize":         "4c",
        "use_psr_drape": False,
    },
    "Malapert Massif": {
        # 0-60° avoids the 0/360 seam that caused GMT status 72.
        "region":        [0, 60, -88.0, -83.0],
        "azimuth":       200,
        "elev_angle":    25,
        "zsize":         "3c",
        "use_psr_drape": False,
    },
}


class DataManager:
    """Loads, calibrates, aligns, and computes derived layers from the heightmap and illumination datasets."""

    def __init__(self):
        print("[DataManager] Loading heightmap ...", flush=True)
        self.hmap_m = self._load_heightmap()

        print("[DataManager] Loading illumination ...", flush=True)
        self.illum_rgb = self._load_illumination()
        self.illum_lum = self._to_luminance(self.illum_rgb)  # 0-255 float32

        print("[DataManager] Aligning resolutions ...", flush=True)
        target = self.illum_lum.shape
        if self.hmap_m.shape != target:
            zy = target[0] / self.hmap_m.shape[0]
            zx = target[1] / self.hmap_m.shape[1]
            self.hmap_m = scipy_zoom(self.hmap_m, (zy, zx), order=1)

        print("[DataManager] Computing derived layers ...", flush=True)
        self.slope_deg = self._compute_slope(self.hmap_m)
        self.psr_mask  = self._compute_psr(self.illum_lum, threshold=50)
        self.safety    = self._compute_safety()

        self.H, self.W = self.hmap_m.shape
        print(f"[DataManager] Ready. Grid: {self.W}x{self.H}", flush=True)

    def _load_heightmap(self):
        """Load ldem_4.tif and return elevation in metres."""
        arr = np.array(Image.open(HEIGHTMAP_PATH))
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            elev_m = arr.astype(np.float32) * 1000.0
        elif arr.dtype in (np.uint16, np.int16):
            elev_m = arr.astype(np.float32) * 0.5 - 8000.0
        else:
            elev_m = (arr.astype(np.float32) / 255.0
                      * (ELEV_MAX_M - ELEV_MIN_M) + ELEV_MIN_M)
        return elev_m

    def _load_illumination(self):
        img = Image.open(ILLUMINATION_PATH).convert("RGB")
        return np.array(img).astype(np.float32)

    def _to_luminance(self, rgb):
        # ITU-R BT.601 perceptual luminance weights
        return (0.299 * rgb[..., 0]
              + 0.587 * rgb[..., 1]
              + 0.114 * rgb[..., 2]).astype(np.float32)

    def _compute_slope(self, elev_m):
        """Compute slope in degrees from elevation gradient with latitude correction.

        East-west pixel spacing shrinks as cos(latitude); without correction polar slopes appear ~22x flatter than they are.
        """
        H, W = elev_m.shape
        gy, gx = np.gradient(elev_m)   # metres per pixel, uncorrected

        lats = np.linspace(LAT_MAX, LAT_MIN, H, dtype=np.float32)[:, np.newaxis]
        cos_lat = np.cos(np.radians(lats)).clip(min=1e-6)  # avoid div-by-zero at poles

        res_x = GROUND_RES_M_PER_PX * cos_lat
        res_y = GROUND_RES_M_PER_PX

        slope_mpm = np.sqrt((gx / res_x) ** 2 + (gy / res_y) ** 2)
        return np.degrees(np.arctan(slope_mpm)).astype(np.float32)

    def _compute_psr(self, lum, threshold=50):
        """Threshold illumination to derive Permanently Shadowed Regions."""
        return (lum < threshold).astype(bool)

    def _compute_safety(self):
        """Composite landing safety score (0=unsafe, 1=ideal) weighted by flatness, sunlight, and elevation."""
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
        """Crop all arrays to a geographic bounding box. region: [lon0, lon1, lat0, lat1]"""
        lon0, lon1, lat0, lat1 = region
        r0 = int((LAT_MAX - lat1) / (LAT_MAX - LAT_MIN) * self.H)
        r1 = int((LAT_MAX - lat0) / (LAT_MAX - LAT_MIN) * self.H)
        c0 = int((lon0 - LON_MIN)  / (LON_MAX - LON_MIN) * self.W)
        c1 = int((lon1 - LON_MIN)  / (LON_MAX - LON_MIN) * self.W)
        r0, r1 = max(r0, 0), min(r1, self.H)
        c0, c1 = max(c0, 0), min(c1, self.W)
        if c0 >= c1:  # handle lon wrapping for full-pole view (0-360)
            c0, c1 = 0, self.W
        sl = (slice(r0, r1), slice(c0, c1))

        raw = {
            "hmap_m":    self.hmap_m[sl],
            "illum_lum": self.illum_lum[sl],
            "slope_deg": self.slope_deg[sl],
            "psr_mask":  self.psr_mask[sl],
            "safety":    self.safety[sl],
        }

        # Upsample polar crops that have too few rows to avoid stripe artefacts.
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
        """Wrap a cropped numpy array as xarray.DataArray for PyGMT.

        Sets spacing explicitly so GMT does not infer it from a sparse coordinate array.
        """
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
                # Explicit spacing prevents GMT from inferring a wrong increment
                # from a coarse or upsampled crop.
                "spacing": float((lon1 - lon0) / max(cols - 1, 1)),
            },
        )


def _get_cmap(name):
    """Return a matplotlib colormap by name, preferring cmcrameri if installed."""
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
        "illum":   "plasma",
        "cividis": "cividis",
    }
    return plt.get_cmap(fallbacks.get(name, "viridis"))


def array_to_rgba(arr2d, cmap_name="batlow", vmin=None, vmax=None,
                  flip_y=True):
    """Convert a 2D float array to a uint32 RGBA array for Bokeh image_rgba.

    Bokeh requires uint32 packed as R|G<<8|B<<16|A<<24. flip_y=True corrects for arrays with origin at the top.
    """
    vmin = vmin if vmin is not None else float(arr2d.min())
    vmax = vmax if vmax is not None else float(arr2d.max())
    norm  = np.clip((arr2d - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    cm    = _get_cmap(cmap_name)
    rgba_f  = cm(norm)                          # H x W x 4 float [0,1]
    rgba_u8 = (rgba_f * 255).astype(np.uint8)
    packed  = (  rgba_u8[..., 0].astype(np.uint32)
              | (rgba_u8[..., 1].astype(np.uint32) << 8)
              | (rgba_u8[..., 2].astype(np.uint32) << 16)
              | (rgba_u8[..., 3].astype(np.uint32) << 24))
    if flip_y:
        packed = np.flipud(packed)
    return packed


def _bokeh_palette(cmap_name, n=256):
    """Convert a matplotlib/cmcrameri colormap to a Bokeh hex palette list."""
    cm = _get_cmap(cmap_name)
    return [
        "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        for r, g, b, _ in (cm(i / (n - 1)) for i in range(n))
    ]


def _prepare_illum_for_display(illum_arr, psr_mask, target_width=2048):
    """Gaussian-smooth, bicubic-upsample, then clamp shadowed pixels to restore sharp PSR boundaries."""
    rows, cols = illum_arr.shape

    smoothed = scipy_gaussian(illum_arr.astype(np.float32), sigma=0.8)

    if cols < target_width:
        scale_factor = target_width / cols
        upsampled = scipy_zoom(smoothed, scale_factor, order=3)
        psr_up = scipy_zoom(
            psr_mask.astype(np.float32), scale_factor, order=0
        ).astype(bool)
    else:
        upsampled = smoothed
        psr_up    = psr_mask

    non_psr_vals = upsampled[~psr_up]
    if len(non_psr_vals) > 0:
        psr_dark = float(np.percentile(non_psr_vals, 5))
    else:
        psr_dark = 0.0
    upsampled = np.where(psr_up, np.minimum(upsampled, psr_dark), upsampled)

    return upsampled.astype(np.float32)


import math as _disc_math

_ILLUM_DISC_LAT_MIN = -90.0   # disc centre (south pole)
_ILLUM_DISC_LAT_MAX = -83.0   # disc outer edge
_ILLUM_DISC_R_MAX   = 10.0    # disc radius in Bokeh plot units


def _illum_lonlat_to_disc(lons_deg, lats_deg):
    """Convert (longitude, latitude) arrays to azimuthal equidistant disc coordinates.

        r = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * R_MAX
        x = r * sin(lon_rad),  y = r * cos(lon_rad)
    """
    lons = np.asarray(lons_deg, dtype=float)
    lats = np.asarray(lats_deg, dtype=float)
    r     = ((lats - _ILLUM_DISC_LAT_MIN)
             / (_ILLUM_DISC_LAT_MAX - _ILLUM_DISC_LAT_MIN)
             * _ILLUM_DISC_R_MAX)
    theta = lons * _disc_math.pi / 180.0
    xs    = r * np.sin(theta)
    ys    = r * np.cos(theta)
    return xs, ys


def _illum_disc_to_lonlat(x, y):
    """Inverse of _illum_lonlat_to_disc. Returns (lon_deg, lat_deg) or None if outside the disc."""
    r = _disc_math.sqrt(x * x + y * y)
    if r > _ILLUM_DISC_R_MAX:
        return None
    lat = (_ILLUM_DISC_LAT_MIN
           + (r / _ILLUM_DISC_R_MAX)
           * (_ILLUM_DISC_LAT_MAX - _ILLUM_DISC_LAT_MIN))
    theta = _disc_math.atan2(x, y)
    lon   = _disc_math.degrees(theta) % 360.0
    return lon, lat


def _reproject_to_illum_disc(img_rgba_u32, region, disc_px=700):
    """Remap an equirectangular packed-uint32 RGBA image into a square polar disc pixel grid using azimuthal equidistant projection."""
    from scipy.ndimage import map_coordinates as _map_coords
    lon0, lon1, lat0, lat1 = region
    H, W = img_rgba_u32.shape

    # Unpack uint32 RGBA into four float32 channels
    r_ch = ( img_rgba_u32        & 0xFF).astype(np.float32)
    g_ch = ((img_rgba_u32 >>  8) & 0xFF).astype(np.float32)
    b_ch = ((img_rgba_u32 >> 16) & 0xFF).astype(np.float32)
    a_ch = ((img_rgba_u32 >> 24) & 0xFF).astype(np.float32)

    # img_rgba_u32 produced with flip_y=True: row 0 = lat0 (south edge).
    half = disc_px / 2.0
    py_idx, px_idx = np.mgrid[0:disc_px, 0:disc_px]
    px_plot = (px_idx - half) / half * _ILLUM_DISC_R_MAX
    py_plot = (py_idx - half) / half * _ILLUM_DISC_R_MAX

    r_sq   = px_plot**2 + py_plot**2
    inside = r_sq <= _ILLUM_DISC_R_MAX**2

    r_val = np.where(inside, np.sqrt(r_sq), 0.0)
    lat   = np.where(
        inside,
        _ILLUM_DISC_LAT_MIN + (r_val / _ILLUM_DISC_R_MAX)
        * (_ILLUM_DISC_LAT_MAX - _ILLUM_DISC_LAT_MIN),
        np.nan,
    )
    theta = np.arctan2(px_plot, py_plot)
    lon   = np.where(inside, np.degrees(theta) % 360.0, np.nan)

    src_col = (lon  - lon0) / (lon1 - lon0) * (W - 1)
    src_row = (lat  - lat0) / (lat1 - lat0) * (H - 1)
    src_col = np.clip(src_col, 0, W - 1)
    src_row = np.clip(src_row, 0, H - 1)

    coords = [src_row.ravel(), src_col.ravel()]
    out_r = _map_coords(r_ch, coords, order=1, mode="nearest").reshape(disc_px, disc_px)
    out_g = _map_coords(g_ch, coords, order=1, mode="nearest").reshape(disc_px, disc_px)
    out_b = _map_coords(b_ch, coords, order=1, mode="nearest").reshape(disc_px, disc_px)
    out_a = _map_coords(a_ch, coords, order=1, mode="nearest").reshape(disc_px, disc_px)

    out_r = np.where(inside, out_r, 0.0)
    out_g = np.where(inside, out_g, 0.0)
    out_b = np.where(inside, out_b, 0.0)
    out_a = np.where(inside, out_a, 0.0)

    return (  out_r.astype(np.uint32)
            | (out_g.astype(np.uint32) <<  8)
            | (out_b.astype(np.uint32) << 16)
            | (out_a.astype(np.uint32) << 24))


def _physical_aspect_y(lon_span, lat_span, lat0, lat1):
    """Compute the y:x aspect ratio for the Plotly 3D scene using lunar surface metres to avoid polar distortion."""
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
    """Compute a Bokeh figure height so that one pixel equals the same physical distance on both axes."""
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


class RenderEngine2:
    """Produces polar disc PNGs using PyGMT grdview with an azimuthal equidistant projection."""

    @staticmethod
    def render_polar_disc(grid_xr, region, cmap="SCM/batlow",
                          azimuth=200, elev_angle=25, zsize="2c",
                          show_as_illumination=False, illum_grid_xr=None):
        """Render the south pole as a 3D perspective disc using grdview with projection E0/-90/20c."""
        lon0, lon1, lat0, lat1 = region
        lat0 = max(lat0, -89.9)
        lat1 = min(lat1,  89.9)

        vmin = float(np.nanmin(grid_xr.values))
        vmax = float(np.nanmax(grid_xr.values))
        inc  = max((vmax - vmin) / 100.0, 1.0)

        tmp = tempfile.NamedTemporaryFile(suffix="_disc.png", delete=False)
        out = tmp.name
        tmp.close()

        fig = pygmt.Figure()

        if show_as_illumination and illum_grid_xr is not None:
            pygmt.makecpt(cmap="SCM/lajolla", series=[0, 255, 2.55],
                          continuous=True)
        else:
            pygmt.makecpt(cmap=cmap, series=[vmin, vmax, inc], continuous=True)

        # Inline shading string avoids two PyGMT 0.13.0 bugs with DataArray shading args.
        _shading_str = "+a315+nt1"

        _grdview_kwargs = dict(
            grid=grid_xr,
            region=[lon0, lon1, lat0, lat1],
            projection="E0/-90/20c",
            perspective=[azimuth, elev_angle],
            zsize=zsize,
            surftype="s",
            cmap=True,
            shading=_shading_str,
            frame=["xa45", "ya10", "WSnE"],
        )
        if show_as_illumination and illum_grid_xr is not None:
            _grdview_kwargs["drapegrid"] = illum_grid_xr

        with pygmt.config(
            FONT_ANNOT_PRIMARY="10p,Helvetica,white",
            FONT_LABEL="11p,Helvetica,white",
            MAP_FRAME_PEN="0.8p,white",
            COLOR_NAN="gray10",
        ):
            fig.grdview(**_grdview_kwargs)

            if show_as_illumination and illum_grid_xr is not None:
                fig.colorbar(
                    frame=["a50", "x+lSunlight exposure",
                           "y+l(shadow to sunlit)"],
                    position="JBC+o0c/-1.5c+w10c/0.4c+h",
                )
            else:
                fig.colorbar(
                    frame=["a2000", "x+lElevation", "y+lm"],
                    position="JBC+o0c/-1.5c+w10c/0.4c+h",
                )

            fig.savefig(out, dpi=300, crop=True)

        # Remove white background for dark UI
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
    def render_polar_disc_psr(grid_xr, illum_xr, region,
                              azimuth=160, elev_angle=35, zsize="4c"):
        """Render a PSR-composite 3D disc: PSR pixels clamped to 0 (dark), non-PSR stretched to [80, 255] using SCM/oslo."""
        lon0, lon1, lat0, lat1 = region
        lat0 = max(lat0, -89.9)
        lat1 = min(lat1,  89.9)

        illum_arr = illum_xr.values.astype(float)
        psr_mask  = illum_arr < 1.0
        drape_arr = np.where(
            psr_mask,
            0.0,
            80.0 + (illum_arr / 255.0) * (255.0 - 80.0),
        )
        drape_xr = xr.DataArray(
            drape_arr,
            coords=illum_xr.coords,
            dims=illum_xr.dims,
            attrs={"long_name": "psr_composite", "units": "luminance"},
        )
        # Resample drapegrid to match elevation grid coordinates exactly (prevents GMT status 79).
        drape_xr = drape_xr.interp(
            lat=grid_xr.coords["lat"],
            lon=grid_xr.coords["lon"],
            method="linear",
        )

        tmp = tempfile.NamedTemporaryFile(suffix="_psr.png", delete=False)
        out = tmp.name
        tmp.close()

        fig = pygmt.Figure()
        pygmt.makecpt(cmap="SCM/oslo", series=[0, 255, 2.55], continuous=True)

        with pygmt.config(
            FONT_ANNOT_PRIMARY="10p,Helvetica,white",
            FONT_LABEL="11p,Helvetica,white",
            MAP_FRAME_PEN="0.8p,white",
            COLOR_NAN="gray10",
        ):
            fig.grdview(
                grid=grid_xr,
                region=[lon0, lon1, lat0, lat1],
                # Mercator for sub-region site cards; azimuthal equidistant is only correct for full 360° polar views.
                projection="M12c",
                perspective=[azimuth, elev_angle],
                zsize=zsize,
                surftype="i600",
                cmap=True,
                shading="+a315+ne0.6",
                drapegrid=drape_xr,
                frame=["WSnE"],
            )
            fig.savefig(out, dpi=150, crop=True)

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
    def render_tour_snapshot(grid_xr, illum_xr, region,
                             azimuth=160, elev_angle=35, zsize="3c",
                             use_psr_drape=False):
        """Fast 3D site-card render: 12c canvas, 150 DPI, tight sub-region."""
        if use_psr_drape:
            return RenderEngine2.render_polar_disc_psr(
                grid_xr, illum_xr, region,
                azimuth=azimuth, elev_angle=elev_angle, zsize=zsize,
            )

        lon0, lon1, lat0, lat1 = region
        lat0 = max(lat0, -89.9)
        lat1 = min(lat1,  89.9)

        vmin = float(np.nanmin(grid_xr.values))
        vmax = float(np.nanmax(grid_xr.values))
        inc  = max((vmax - vmin) / 100.0, 1.0)

        tmp = tempfile.NamedTemporaryFile(suffix="_snap.png", delete=False)
        out = tmp.name
        tmp.close()

        fig = pygmt.Figure()
        pygmt.makecpt(cmap="SCM/batlow", series=[vmin, vmax, inc],
                      continuous=True)

        with pygmt.config(
            FONT_ANNOT_PRIMARY="10p,Helvetica,white",
            FONT_LABEL="11p,Helvetica,white",
            MAP_FRAME_PEN="0.8p,white",
            COLOR_NAN="gray10",
        ):
            fig.grdview(
                grid=grid_xr,
                region=[lon0, lon1, lat0, lat1],
                projection="M12c",
                perspective=[azimuth, elev_angle],
                zsize=zsize,
                surftype="i600",
                cmap=True,
                shading="+a315+ne0.6",
                frame=["WSnE"],
            )
            fig.savefig(out, dpi=150, crop=True)

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


class AppState2(param.Parameterized):
    """Reactive state for the public-facing dashboard."""

    azimuth           = param.Integer(default=200, bounds=(0,   360))
    elev_angle        = param.Integer(default=25,  bounds=(10,  80))
    vert_exag         = param.Selector(default="2c",
                                       objects=["1c", "2c", "3c", "4c", "5c"])
    cmap_name         = param.Selector(default="batlow",
                                       objects=CMAP_OPTIONS)
    illum_cmap_name   = param.Selector(default="cividis",
                                       objects=ILLUM_CMAP_OPTIONS)
    selected_site     = param.Selector(default=list(ARTEMIS_SITES.keys())[0],
                                       objects=list(ARTEMIS_SITES.keys()))


def build_illumination_map(dm, state, info_pane, bar_highlighter=None,
                           info_style=""):
    """Build an interactive polar disc showing solar illumination for the south pole."""
    region = ILLUM_DISC_CROP_REGION
    lon0, lon1, lat0, lat1 = region
    crop = dm.crop(region)

    _rows_total = crop["illum_lum"].shape[0]
    _full_lat_span = abs(lat1 - lat0)
    _polar_start_row = int(_rows_total * (abs(lat1 - (-80.0)) / _full_lat_span))
    _polar_start_row = max(0, min(_polar_start_row, _rows_total - 1))
    _polar_slice = crop["illum_lum"][_polar_start_row:, :]
    _illum_flat  = _polar_slice.flatten()
    _vmin_actual = float(np.percentile(_illum_flat, 1))
    _vmax_actual = float(np.percentile(_illum_flat, 99))

    _illum_display = _prepare_illum_for_display(
        crop["illum_lum"], crop["psr_mask"], target_width=4096,
    )
    _polar_start_recalc = int(
        _illum_display.shape[0] * (abs(lat1 - (-80.0)) / _full_lat_span)
    )
    _polar_start_recalc = max(0, min(_polar_start_recalc,
                                     _illum_display.shape[0] - 1))
    _polar_recalc_flat = _illum_display[_polar_start_recalc:, :].flatten()
    _vmin_display = float(np.percentile(_polar_recalc_flat, 1))
    _vmax_display = float(np.percentile(_polar_recalc_flat, 99))

    _DISC_PX = 660   # output square size in pixels

    def _make_illum_disc_img(cmap_name):
        eq_img = array_to_rgba(
            _illum_display,
            cmap_name=cmap_name,
            vmin=_vmin_display,
            vmax=_vmax_display,
            flip_y=True,
        )
        return _reproject_to_illum_disc(eq_img, region, disc_px=_DISC_PX)

    disc_img = _make_illum_disc_img(state.illum_cmap_name)

    _sd_rows, _sd_cols = _illum_display.shape
    _slope_raw = crop["slope_deg"]
    if _slope_raw.shape != (_sd_rows, _sd_cols):
        _slope_up = scipy_zoom(
            _slope_raw.astype(np.float32),
            (_sd_rows / _slope_raw.shape[0], _sd_cols / _slope_raw.shape[1]),
            order=1,
        )
    else:
        _slope_up = _slope_raw.astype(np.float32)

    _safe_mask   = _slope_up <= 5.0
    _warn_mask   = (_slope_up > 5.0) & (_slope_up <= 15.0)
    _danger_mask =  _slope_up > 15.0

    _slope_rgba = np.zeros((_sd_rows, _sd_cols, 4), dtype=np.uint8)
    _slope_rgba[_safe_mask,   0] = 34;  _slope_rgba[_safe_mask,   1] = 197
    _slope_rgba[_safe_mask,   2] = 94;  _slope_rgba[_safe_mask,   3] = 55
    _slope_rgba[_warn_mask,   0] = 251; _slope_rgba[_warn_mask,   1] = 191
    _slope_rgba[_warn_mask,   2] = 36;  _slope_rgba[_warn_mask,   3] = 210
    _slope_rgba[_danger_mask, 0] = 239; _slope_rgba[_danger_mask, 1] = 68
    _slope_rgba[_danger_mask, 2] = 68;  _slope_rgba[_danger_mask, 3] = 230

    _slope_eq = np.flipud(
          _slope_rgba[..., 0].astype(np.uint32)
        | (_slope_rgba[..., 1].astype(np.uint32) <<  8)
        | (_slope_rgba[..., 2].astype(np.uint32) << 16)
        | (_slope_rgba[..., 3].astype(np.uint32) << 24)
    )
    slope_disc_img = _reproject_to_illum_disc(
        _slope_eq, region, disc_px=_DISC_PX
    )

    _margin = 0.6
    _ax_range = (-_ILLUM_DISC_R_MAX - _margin, _ILLUM_DISC_R_MAX + _margin)

    disc_source  = ColumnDataSource(dict(
        image=[disc_img],
        x=[-_ILLUM_DISC_R_MAX], y=[-_ILLUM_DISC_R_MAX],
        dw=[2 * _ILLUM_DISC_R_MAX], dh=[2 * _ILLUM_DISC_R_MAX],
    ))
    slope_source = ColumnDataSource(dict(
        image=[slope_disc_img],
        x=[-_ILLUM_DISC_R_MAX], y=[-_ILLUM_DISC_R_MAX],
        dw=[2 * _ILLUM_DISC_R_MAX], dh=[2 * _ILLUM_DISC_R_MAX],
    ))

    p = figure(
        width=680, height=680,
        x_range=_ax_range,
        y_range=_ax_range,
        title="",
        tools="wheel_zoom,pan,reset,save,tap",
        active_scroll="wheel_zoom",
        match_aspect=True,
        aspect_scale=1.0,
        background_fill_color="#000010",
        border_fill_color="#0a0a14",
    )
    p.xaxis.major_label_text_font_size = "0pt"
    p.yaxis.major_label_text_font_size = "0pt"
    p.xaxis.major_tick_line_color = None
    p.yaxis.major_tick_line_color = None
    p.xaxis.minor_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.xaxis.axis_line_color = None
    p.yaxis.axis_line_color = None
    p.xaxis.axis_label = ""
    p.yaxis.axis_label = ""
    p.min_border_left   = 10
    p.min_border_right  = 10
    p.min_border_top    = 10
    p.min_border_bottom = 10

    _theta_ring = np.linspace(0, 2 * _disc_math.pi, 360)
    p.line(
        x=(_ILLUM_DISC_R_MAX * np.sin(_theta_ring)).tolist(),
        y=(_ILLUM_DISC_R_MAX * np.cos(_theta_ring)).tolist(),
        line_color="#475569", line_width=1.2, line_alpha=0.8,
    )
    p.scatter(x=[0], y=[0], marker="cross", size=8,
              line_color="#94a3b8", line_width=1.5)
    for _ref_lat in [-85.0, -88.0]:
        _ref_r = ((_ref_lat - _ILLUM_DISC_LAT_MIN)
                  / (_ILLUM_DISC_LAT_MAX - _ILLUM_DISC_LAT_MIN)
                  * _ILLUM_DISC_R_MAX)
        p.line(
            x=(_ref_r * np.sin(_theta_ring)).tolist(),
            y=(_ref_r * np.cos(_theta_ring)).tolist(),
            line_color="#1e3a5f", line_width=0.6, line_alpha=0.6,
            line_dash="dashed",
        )
    _lat_label_xs, _lat_label_ys, _lat_label_texts = [], [], []
    for _ref_lat, _label in [(-85.0, "−85°"), (-88.0, "−88°")]:
        _ref_r = ((_ref_lat - _ILLUM_DISC_LAT_MIN)
                  / (_ILLUM_DISC_LAT_MAX - _ILLUM_DISC_LAT_MIN)
                  * _ILLUM_DISC_R_MAX)
        _lat_label_xs.append(_ref_r * 0.707)
        _lat_label_ys.append(_ref_r * 0.707)
        _lat_label_texts.append(_label)
    _lat_label_source = ColumnDataSource(dict(
        x=_lat_label_xs, y=_lat_label_ys, text=_lat_label_texts,
    ))
    p.add_layout(LabelSet(
        x="x", y="y", text="text", source=_lat_label_source,
        text_color="#475569", text_font_size="8px",
    ))

    p.image_rgba(source=disc_source, image="image",
                 x="x", y="y", dw="dw", dh="dh")

    _slope_renderer = p.image_rgba(
        source=slope_source, image="image",
        x="x", y="y", dw="dw", dh="dh",
        visible=False,
    )

    _vmin_pct = _vmin_actual / 255.0 * 100.0
    _vmax_pct = _vmax_actual / 255.0 * 100.0
    _illum_mapper = LinearColorMapper(
        palette=_bokeh_palette(state.illum_cmap_name),
        low=_vmin_pct, high=_vmax_pct,
    )

    site_lons  = [v[0] for v in ARTEMIS_SITES.values()]
    site_lats  = [v[1] for v in ARTEMIS_SITES.values()]
    site_names = list(ARTEMIS_SITES.keys())
    _valid_idx = [i for i, lat in enumerate(site_lats)
                  if _ILLUM_DISC_LAT_MIN <= lat <= _ILLUM_DISC_LAT_MAX]
    _s_lons = [site_lons[i] for i in _valid_idx]
    _s_lats = [site_lats[i] for i in _valid_idx]
    _s_names = [site_names[i] for i in _valid_idx]

    _sx, _sy = _illum_lonlat_to_disc(_s_lons, _s_lats)
    site_source = ColumnDataSource(dict(
        x=_sx.tolist(), y=_sy.tolist(), name=_s_names,
    ))
    p.scatter(
        x="x", y="y", source=site_source,
        marker="star", size=12,
        fill_color="#facc15", line_color="#ffffff", line_width=1.0,
        fill_alpha=0.95,
    )
    p.add_layout(LabelSet(
        x="x", y="y", text="name", source=site_source,
        text_color="#facc15", text_font_size="8px",
        text_font_style="bold",
        text_outline_color="#000000",
        x_offset=6, y_offset=4,
    ))

    _s_icons = [SITE_ICONS.get(n, "☀") for n in _s_names]
    icon_source = ColumnDataSource(dict(
        x=_sx.tolist(), y=_sy.tolist(), icon=_s_icons,
    ))
    from bokeh.models import Text as BokehText
    p.add_glyph(icon_source, BokehText(
        x="x", y="y", text="icon",
        text_color="#ffffff", text_font_size="11px",
        text_align="center", text_baseline="bottom",
        x_offset=0, y_offset=18,
    ))

    def on_tap(event):
        result = _illum_disc_to_lonlat(event.x, event.y)
        if result is None:
            return
        lon, lat = result
        best_name = None
        best_dist = float("inf")
        for name, (slon, slat, _desc) in ARTEMIS_SITES.items():
            dist = (lon - slon)**2 + (lat - slat)**2
            if dist < best_dist:
                best_dist = dist
                best_name = name
        if best_name and best_dist < 200:
            _slon, _slat, _desc = ARTEMIS_SITES[best_name]
            res = dm.sample_point(_slon, _slat)
            _slope = res["slope_deg"]
            if _slope < 5.0:
                _slope_str = (f"<span style='color:#22c55e;font-weight:bold'>"
                              f"✔ {_slope:.1f}° Safe</span>")
            elif _slope <= 15.0:
                _slope_str = (f"<span style='color:#f59e0b;font-weight:bold'>"
                              f"⚠ {_slope:.1f}° Caution</span>")
            else:
                _slope_str = (f"<span style='color:#ef4444;font-weight:bold'>"
                              f"✖ {_slope:.1f}° Unsafe</span>")
            _illum = res["illumination"]
            _illum_str = (
                f"<span style='color:#facc15;font-weight:bold'>"
                f"{_illum:.0f}% ☀</span>"
                if _illum >= 70 else f"{_illum:.0f}%"
            )
            _icon = SITE_ICONS.get(best_name, "☀")
            _role_labels = {
                "❄": "Water ice access site",
                "☀": "Solar power peak site",
                "⬡": "Communications relay site",
            }
            _role = _role_labels.get(_icon, "Candidate site")
            info_pane.object = (
                f"<div style='{info_style}'>"
                f"<h3 style='color:#facc15;margin:0 0 4px 0;"
                f"font-size:15px;font-weight:600;'>{_icon} {best_name}</h3>"
                f"<p style='margin:0 0 8px 0;color:#64748b;"
                f"font-size:11px;font-style:italic;'>{_role}</p>"
                f"<p style='margin:0 0 10px 0;color:#94a3b8;'>{_desc}</p>"
                f"<p style='margin:0;'>"
                f"<strong>Elevation:</strong> {res['elevation_m']:.0f} m"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"<strong>Illumination:</strong> {_illum_str}"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"<strong>Slope:</strong> {_slope_str}"
                f"</p></div>"
            )
            if bar_highlighter is not None:
                bar_highlighter(best_name)

    p.on_event(Tap, on_tap)

    def update_illum_image(new_cmap_name):
        new_img = _make_illum_disc_img(new_cmap_name)
        disc_source.data = dict(
            image=[new_img],
            x=[-_ILLUM_DISC_R_MAX], y=[-_ILLUM_DISC_R_MAX],
            dw=[2 * _ILLUM_DISC_R_MAX], dh=[2 * _ILLUM_DISC_R_MAX],
        )
        _illum_mapper.palette = _bokeh_palette(new_cmap_name)

    def pan_to(site_lon, site_lat):
        # No-op: disc always shows all sites simultaneously.
        pass

    def set_slope_visible(visible: bool):
        _slope_renderer.visible = visible

    return p, update_illum_image, pan_to, set_slope_visible, \
           state.illum_cmap_name, _vmin_pct, _vmax_pct


def build_site_comparison_chart(dm):
    """Horizontal bar chart comparing peak solar illumination across all 13 Artemis sites.

    Returns (bokeh_figure, highlight_fn) where highlight_fn(site_name) highlights the selected bar.
    """
    _WIN = 0.025   # half-width in degrees
    _rng = np.random.default_rng(seed=42)   # fixed seed for reproducibility
    names_raw = list(ARTEMIS_SITES.keys())
    illum_vals_raw = []
    for name in names_raw:
        slon, slat, _ = ARTEMIS_SITES[name]
        _lons = _rng.uniform(slon - _WIN, slon + _WIN, size=50)
        _lats = _rng.uniform(slat - _WIN, slat + _WIN, size=50)
        _vals = [
            dm.sample_point(float(lo), float(la))["illumination"]
            for lo, la in zip(_lons, _lats)
        ]
        illum_vals_raw.append(max(_vals))

    # Sort ascending so Bokeh places the best-illuminated site at the top.
    _sorted = sorted(zip(names_raw, illum_vals_raw), key=lambda x: x[1])
    names      = [p[0] for p in _sorted]
    illum_vals = [p[1] for p in _sorted]
    colours    = ["#facc15"] * len(names)
    short_names = [
        SITE_ICONS.get(n, "☀") + " " + (n if len(n) <= 20 else n[:18] + "…")
        for n in names
    ]

    source = ColumnDataSource(dict(
        names=names,
        illum=illum_vals,
        color=colours,
        short=short_names,
    ))

    p = figure(
        y_range=short_names,
        width=700,
        height=380,
        title=(
            "Peak Solar Illumination by Candidate Site  "
            "(70% = minimum for a solar-powered base)"
        ),
        tools="",
        toolbar_location=None,
        background_fill_color="#111827",
        border_fill_color="#0a0a14",
    )

    p.title.text_color            = "#e0e0e0"
    p.title.text_font_size        = "13px"
    p.xaxis.axis_label            = "Peak illumination within landing zone (%)"
    p.xaxis.axis_label_text_color = "#b0b0b0"
    p.xaxis.major_label_text_color = "#b0b0b0"
    p.yaxis.major_label_text_color = "#d1d5db"
    p.yaxis.major_label_text_font_size = "11px"
    p.xgrid.grid_line_color       = "#1e293b"
    p.ygrid.grid_line_color       = None
    p.outline_line_color          = None

    p.hbar(
        y="short", right="illum", height=0.7,
        source=source,
        color="color",
        line_color=None,
    )

    threshold_line = Span(
        location=70,
        dimension="height",
        line_color="#ef4444",
        line_dash="dashed",
        line_width=1.5,
        line_alpha=0.8,
    )
    p.add_layout(threshold_line)

    p.add_layout(Label(
        x=71, y=0.3,
        text="70% survival threshold",
        text_color="#ef4444",
        text_font_size="10px",
        x_units="data", y_units="data",
    ))

    p.x_range.start = 0
    p.x_range.end   = 105

    def highlight(site_name):
        """Set the selected site bar to white; all others gold."""
        _icon = SITE_ICONS.get(site_name, "☀")
        _base = site_name if len(site_name) <= 20 else site_name[:18] + "…"
        short_site = _icon + " " + _base
        new_colours = [
            "#ffffff" if s == short_site else "#facc15"
            for s in source.data["short"]
        ]
        source.patch({"color": [(slice(None), new_colours)]})

    return p, highlight


def create_app2():
    """Factory for the Panel application. Returns a Panel template ready for pn.serve()."""
    state = AppState2()

    _INFO_STYLE = (
        "color:#d0d8e8;background:#111827;padding:12px;"
        "border-radius:6px;font-size:13px;min-height:100px;"
        "line-height:1.7;font-family:inherit;"
    )

    _TILT_HTML = """
        <div style="background:#0f172a; border: 1px solid #334155; border-radius:12px; padding:20px;
                    color:#e2e8f0; font-family: sans-serif; max-width:550px; line-height:1.6;">

        <p style="margin:0 0 12px 0; font-weight:700; font-size:15px; color:#facc15; text-transform:uppercase; letter-spacing:0.5px;">
            The Physics of "Cold Traps"
        </p>

        <p style="margin:0 0 15px 0; font-size:13px; color:#94a3b8;">
            Because the Moon's axis is tilted only <strong style="color:#e2e8f0;">1.5°</strong>, sunlight at the poles arrives at a grazing angle, perpetually skimming the horizon.
        </p>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-bottom:10px;">
            <div style="background:#1e293b; padding:12px; border-radius:8px; border-left:4px solid #facc15;">
            <strong style="color:#facc15; display:block; margin-bottom:4px; font-size:12px;">PEAKS OF LIGHT</strong>
            <span style="font-size:12px; color:#cbd5e1;">High ridges catch the sun above the horizon, providing near-constant solar power for habitats.</span>
            </div>
            <div style="background:#1e293b; padding:12px; border-radius:8px; border-left:4px solid #3b82f6;">
            <strong style="color:#7dd3fc; display:block; margin-bottom:4px; font-size:12px;">COLD TRAPS</strong>
            <span style="font-size:12px; color:#cbd5e1;">Deep crater floors stay in 1-billion-year shadows, preserving ice.</span>
            </div>
        </div>

        <p style="margin:0; font-size:11px; color:#64748b; font-style:italic; text-align:center;">
            A safe landing requires finding a "Golden Zone" where these two extremes meet.
        </p>
        </div>
        """
    tilt_pane = pn.pane.HTML(_TILT_HTML, sizing_mode="stretch_width")

    intro_and = pn.pane.Markdown(
        "### The New Frontier: Why the South Pole?\n\n"
        "**Humanity is returning to the Moon, this time to stay.**\n\n"
        "### AND: A Land of Infinite Resources\n\n"
        "NASA's **Artemis III** mission will land the crew at the lunar "
        "south pole, a region of extreme strategic value. Unlike the dusty "
        "plains explored by Apollo, this landscape holds two keys "
        "to our future: **water ice** frozen in permanent shadow and **peaks "
        "of eternal light** where the sun almost never sets. These resources "
        "provide the water, oxygen, and near-infinite solar power needed for "
        "long-term lunar habitation.",
        styles={
            "color": "#e2e8f0",
            "background": "#1e3a5f",
            "padding": "14px",
            "border-radius": "8px",
            "font-size": "13px",
            "line-height": "1.8",
            "border-left": "4px solid #3b82f6",
        },
    )

    intro_but = pn.pane.Markdown(
        "### BUT: A Landscape of Extreme Hostility\n\n"
        "The south pole has rugged terrain"
        "to navigate. Craters plunge **4 km deep**, and because the sun never "
        "rises more than 12° above the horizon, some crater floors have been "
        "in total darkness for **billions of years**. To land safely, a site "
        "must be **flat** (ideally much than a 15° slope), **constantly sunlit** for "
        "solar power, and **within reach of ice**. These three "
        "almost never coincide.",
        styles={
            "color": "#e2e8f0",
            "background": "#3b1f1f",
            "padding": "14px",
            "border-radius": "8px",
            "font-size": "13px",
            "line-height": "1.8",
            "border-left": "4px solid #ef4444",
        },
    )

    intro_therefore = pn.pane.Markdown(
        "### THEREFORE: We Must Find the 'Golden Zones'\n\n"
        "NASA has identified **13 candidate landing regions** where these "
        "survival constraints overlap. Three specific sites define the unique "
        "trade-offs of the pole:\n\n"
        "**Peak near Shackleton**: A ridge sitting just 1° from the pole. "
        "It receives **70%+ annual sunlight** in certain spots, serving as the perfect "
        "solar power station overlooking the icy depths of Shackleton crater.\n\n"
        "**Connecting Ridge**: A high-altitude 'bridge' that provides the "
        "rare combination of **consistent moderate/high and flat terrain**, ideal "
        "for long-duration rover operations.\n\n"
        "**Malapert Massif**: A massive mountain (7000+ m elevation) with an **unobstructed "
        "view of Earth**."
        "\n\n"
        "**Use the Guided Tour and interactive map below to explore the "
        "strategic landscape of all 13 sites.**",
        styles={
            "color": "#e2e8f0",
            "background": "#1a3a1a",
            "padding": "14px",
            "border-radius": "8px",
            "font-size": "13px",
            "line-height": "1.8",
            "border-left": "4px solid #22c55e",
        },
    )

    info_pane = pn.pane.HTML(
        f"<div style='{_INFO_STYLE}'>"
        "<em>Click a gold star on the map to learn about a candidate "
        "landing site.</em></div>",
        width=700,
    )

    site_card_pane = pn.pane.PNG(
        None,
        sizing_mode="scale_width",
        max_width=700,
        visible=False,
    )
    site_card_label = pn.pane.Markdown(
        "",
        styles={"color": "#7dd3fc", "font-size": "12px",
                "margin": "12px 0 4px 0"},
        visible=False,
    )

    disc_pane = pn.pane.PNG(
        None,
        sizing_mode="scale_width",
        max_width=800,
    )

    status_pane = pn.pane.Markdown(
        "**Status:** Ready",
        styles={"color": "#86efac", "font-size": "12px"},
        width=240,
    )

    _bar_ref = {}

    _spinner = pn.indicators.LoadingSpinner(
        value=True,
        width=60, height=60,
        color="warning",
        bgcolor="dark",
    )
    _spinner_label = pn.pane.Markdown(
        "_Preparing illumination map..._",
        styles={"color": "#6b7280", "font-size": "12px"},
    )
    _spinner_start = [__import__("time").time()]

    def _tick_spinner():
        elapsed = int(__import__("time").time() - _spinner_start[0])
        _spinner_label.object = (
            f"_Preparing illumination map... {elapsed}s elapsed_"
        )

    _spinner_cb = pn.state.add_periodic_callback(_tick_spinner, period=1000)

    illum_col = pn.Column(_spinner, _spinner_label)

    _illum_refs = {}

    def _build_illum():
        illum_map, update_fn, pan_fn, slope_fn, \
            _cmap_name, _vmin_pct, _vmax_pct = build_illumination_map(
            DM, state, info_pane,
            bar_highlighter=lambda name: _bar_ref.get("fn", lambda _: None)(name),
            info_style=_INFO_STYLE,
        )
        _illum_refs["update"]    = update_fn
        _illum_refs["pan"]       = pan_fn
        _illum_refs["slope"]     = slope_fn
        _illum_refs["cmap"]      = _cmap_name
        _illum_refs["vmin_pct"]  = _vmin_pct
        _illum_refs["vmax_pct"]  = _vmax_pct
        if _pending_cmap["name"] != state.illum_cmap_name:
            update_fn(_pending_cmap["name"])
        # Schedule UI updates on the Panel event loop (not safe from background thread).
        _bokeh_pane = pn.pane.Bokeh(illum_map)
        pn.state.execute(lambda: illum_col.__setattr__("objects", [_bokeh_pane]))
        pn.state.execute(lambda: _spinner_cb.stop())
        pn.state.execute(_refresh_cbar)

    threading.Thread(target=_build_illum, daemon=True).start()

    _cbar_pane = pn.pane.HTML("", width=70, height=680)

    def _build_cbar_html(cmap_name, vmin_pct, vmax_pct):
        """Build a vertical CSS linear-gradient colourbar as an HTML string."""
        cm = _get_cmap(cmap_name)
        n_stops = 10
        stops = []
        for i in range(n_stops + 1):
            t = i / n_stops
            r, g, b, _ = cm(1.0 - t)
            stops.append(
                f"rgb({int(r*255)},{int(g*255)},{int(b*255)}) {t*100:.0f}%"
            )
        gradient = ", ".join(stops)

        tick_vals = [
            vmax_pct - i * (vmax_pct - vmin_pct) / 4
            for i in range(5)
        ]
        ticks_html = "".join(
            f"<div style='position:absolute;right:28px;"
            f"top:{i*25}%;transform:translateY(-50%);"
            f"font-size:10px;color:#b0b0b0;white-space:nowrap;'>"
            f"{v:.0f}</div>"
            for i, v in enumerate(tick_vals)
        )

        return (
            f"<div style='position:relative;display:inline-block;"
            f"height:300px;width:60px;margin-top:180px;'>"
            f"<div style='position:absolute;left:8px;top:0;width:14px;"
            f"height:100%;background:linear-gradient({gradient});'></div>"
            f"{ticks_html}"
            f"<div style='position:absolute;left:0;top:50%;"
            f"transform:translateX(-50%) translateY(-50%) rotate(-90deg);"
            f"font-size:11px;color:#b0b0b0;white-space:nowrap;"
            f"transform-origin:center center;"
            f"writing-mode:vertical-rl;text-orientation:mixed;"
            f"left:2px;top:0;height:100%;display:flex;"
            f"align-items:center;justify-content:center;'>"
            f"Illumination (%)</div>"
            f"</div>"
        )

    def _refresh_cbar():
        """Update the external colorbar once the map is ready."""
        cmap  = _illum_refs.get("cmap",     state.illum_cmap_name)
        vmin  = _illum_refs.get("vmin_pct", 60.0)
        vmax  = _illum_refs.get("vmax_pct", 75.0)
        _cbar_pane.object = _build_cbar_html(cmap, vmin, vmax)

    _pending_cmap = {"name": state.illum_cmap_name}

    def update_illum_image(cmap):
        _pending_cmap["name"] = cmap
        if "update" in _illum_refs:
            _illum_refs["update"](cmap)

    def pan_illum_to(lon, lat):
        if "pan" in _illum_refs:
            _illum_refs["pan"](lon, lat)

    def set_slope_visible(visible):
        if "slope" in _illum_refs:
            _illum_refs["slope"](visible)

    comparison_chart, highlight_bar = build_site_comparison_chart(DM)
    comparison_pane = pn.pane.Bokeh(comparison_chart)
    _bar_ref["fn"] = highlight_bar

    _render_lock = threading.Lock()

    def trigger_render():
        if not _render_lock.acquire(blocking=False):
            status_pane.object = "**Status:** Rendering..."
            return
        status_pane.object = "**Status:** Rendering disc..."

        def worker():
            try:
                gmt_region = FULL_POLE_REGION_GMT
                gmt_crop   = DM.crop(gmt_region)
                grid_xr = DM.to_xarray(gmt_crop["hmap_m"], gmt_region)

                out = RenderEngine2.render_polar_disc(
                    grid_xr, gmt_region,
                    cmap=f"SCM/{state.cmap_name}",
                    azimuth=state.azimuth,
                    elev_angle=state.elev_angle,
                    zsize=state.vert_exag,
                )
                disc_pane.object = out
                status_pane.object = "**Status:** Done"
            except Exception as exc:
                status_pane.object = f"**Status:** Error — {exc}"
            finally:
                _render_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    render_btn = pn.widgets.Button(
        name="Render Disc View", button_type="primary", width=220,
    )
    render_btn.on_click(lambda _: trigger_render())

    _tour_step = [0]   # list-wrapped for closure mutation

    tour_btn = pn.widgets.Button(
        name="Guided Tour  ▶  Stop 1 / 3",
        button_type="primary",
        width=220,
    )

    def on_tour_click(_):
        if "Restart" in tour_btn.name:
            tour_btn.name = "Guided Tour  ▶  Stop 1 / 3"
        step = _tour_step[0] % len(TOUR_SITES)
        site_name, narrative = TOUR_SITES[step]
        _slon, _slat, site_desc = ARTEMIS_SITES[site_name]
        res = DM.sample_point(_slon, _slat)

        _narrative_html = re.sub(
            r"\*\*(.+?)\*\*", r"<strong>\1</strong>", narrative
        )
        _narrative_html = _narrative_html.replace("\n\n", "</p><p>")

        _t_icon = SITE_ICONS.get(site_name, "☀")
        _t_role_labels = {
            "❄": "Water ice access site",
            "☀": "Solar power peak site",
            "⬡": "Communications relay site",
        }
        _t_role = _t_role_labels.get(_t_icon, "Candidate site")

        _elev_m = res["elevation_m"]
        _pole_avg_m = -2500
        _elev_diff  = _elev_m - _pole_avg_m
        if _elev_diff >= 0:
            _elev_note = (
                f"{_elev_diff:+.0f} m above the south pole average "
                f"— an elevated ridge or plateau"
            )
        else:
            _elev_note = (
                f"{abs(_elev_diff):.0f} m below the south pole average "
                f"— a crater floor or depression"
            )

        # Express illumination as sunlit hours per 708-hr lunar day for audience legibility.
        _illum_pct = res["illumination"]
        _illum_hrs = _illum_pct / 100.0 * 708.0
        _illum_note = (
            f"~{_illum_hrs:.0f} sunlit hours per 708-hr lunar day"
        )

        _slope_val = res["slope_deg"]
        if _slope_val < 5.0:
            _slope_verdict = f"<span style='color:#22c55e'>✔ {_slope_val:.1f}°</span>"
            _slope_note    = "well within the 15° landing limit"
        elif _slope_val <= 15.0:
            _slope_verdict = f"<span style='color:#f59e0b'>⚠ {_slope_val:.1f}°</span>"
            _slope_note    = "approaching the 15° landing limit — caution"
        else:
            _slope_verdict = f"<span style='color:#ef4444'>✖ {_slope_val:.1f}°</span>"
            _slope_note    = "exceeds the 15° landing limit — unsafe"

        info_pane.object = (
            f"<div style='{_INFO_STYLE}'>"
            f"<p style='margin:0 0 4px 0;font-size:11px;"
            f"color:#64748b;font-style:italic;'>{_t_role}</p>"
            f"<p style='margin:0 0 10px 0;'>{_narrative_html}</p>"
            f"<hr style='border:none;border-top:1px solid #1e293b;"
            f"margin:8px 0;'>"
            f"<table style='width:100%;border-collapse:collapse;"
            f"font-size:12px;line-height:1.9;'>"
            f"<tr>"
            f"<td style='color:#64748b;white-space:nowrap;"
            f"padding-right:12px;vertical-align:top;'>"
            f"Height at site</td>"
            f"<td style='color:#e2e8f0;'>"
            f"<strong>{_elev_m:+.0f} m</strong>&nbsp;&nbsp;"
            f"<span style='color:#64748b;font-size:11px;'>{_elev_note}</span>"
            f"</td></tr>"
            f"<tr>"
            f"<td style='color:#64748b;white-space:nowrap;"
            f"padding-right:12px;vertical-align:top;'>"
            f"Sunlight</td>"
            f"<td style='color:#e2e8f0;'>"
            f"<strong>{_illum_pct:.0f}%</strong>&nbsp;&nbsp;"
            f"<span style='color:#64748b;font-size:11px;'>{_illum_note}</span>"
            f"</td></tr>"
            f"<tr>"
            f"<td style='color:#64748b;white-space:nowrap;"
            f"padding-right:12px;vertical-align:top;'>"
            f"Ground steepness</td>"
            f"<td>{_slope_verdict}&nbsp;&nbsp;"
            f"<span style='color:#64748b;font-size:11px;'>{_slope_note}</span>"
            f"</td></tr>"
            f"</table>"
            f"</div>"
        )

        _bar_ref.get("fn", lambda _: None)(site_name)
        pan_illum_to(_slon, _slat)

        _card_params = SITE_CARD_PARAMS.get(site_name)
        if _card_params:
            _icon_label = SITE_ICONS.get(site_name, "☀")
            site_card_label.object  = (
                f"**{_icon_label} Site terrain card — {site_name}**"
            )
            site_card_label.visible = True
            site_card_pane.object   = None
            site_card_pane.visible  = True

            def _card_worker(params=_card_params):
                try:
                    _reg  = params["region"]
                    _crop = DM.crop(_reg)
                    _grid = DM.to_xarray(_crop["hmap_m"], _reg)
                    _illm = DM.to_xarray(
                        _crop["illum_lum"], _reg,
                        name="illumination", units="luminance",
                    )
                    _png = RenderEngine2.render_tour_snapshot(
                        _grid, _illm, _reg,
                        azimuth=params["azimuth"],
                        elev_angle=params["elev_angle"],
                        zsize=params["zsize"],
                        use_psr_drape=params["use_psr_drape"],
                    )
                    # Schedule on Panel event loop (not safe from background thread).
                    pn.state.execute(
                        lambda p=_png: setattr(site_card_pane, "object", p)
                    )
                except Exception as _exc:
                    pn.state.execute(
                        lambda e=_exc: setattr(
                            site_card_label, "object",
                            f"**Card render error:** {e}"
                        )
                    )

            threading.Thread(target=_card_worker, daemon=True).start()
        else:
            site_card_label.visible = False
            site_card_pane.visible  = False

        _tour_step[0] += 1
        if _tour_step[0] % len(TOUR_SITES) == 0:
            tour_btn.name = "Guided Tour  ↺  Restart from Stop 1"
        else:
            next_num = (_tour_step[0] % len(TOUR_SITES)) + 1
            tour_btn.name = f"Guided Tour  ▶  Stop {next_num} / 3"

    tour_btn.on_click(on_tour_click)

    def _make_toggle(base_name, initial_value, button_type, width=220):
        """Create a Toggle widget whose label shows its current state."""
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

    azimuth_sl = pn.widgets.IntSlider(
        name="Rotate view (degrees)", start=0, end=360, step=10,
        value=state.azimuth, width=220,
    )
    tilt_sl = pn.widgets.IntSlider(
        name="Tilt (degrees)", start=10, end=80, step=5,
        value=state.elev_angle, width=220,
    )
    vexag_sl = pn.widgets.Select(
        name="Vertical exaggeration",
        options=["1c", "2c", "3c", "4c", "5c"],
        value=state.vert_exag, width=220,
    )
    cmap_sel = pn.widgets.Select(
        name="Colour scheme", options=CMAP_OPTIONS,
        value=state.cmap_name, width=220,
    )
    illum_cmap_sel = pn.widgets.Select(
        name="Illumination map colours",
        options=ILLUM_CMAP_OPTIONS,
        value=state.illum_cmap_name,
        width=220,
    )

    slope_overlay_tog = _make_toggle("Slope safety zones", False, "warning", width=220)

    def _on_slope_tog(event):
        set_slope_visible(event.new)

    slope_overlay_tog.param.watch(_on_slope_tog, "value")

    def _sync(widget, attr, key):
        def cb(e):
            setattr(state, key, e.new)
        widget.param.watch(cb, attr)

    _sync(azimuth_sl,     "value", "azimuth")
    _sync(tilt_sl,        "value", "elev_angle")
    _sync(vexag_sl,       "value", "vert_exag")
    _sync(cmap_sel,       "value", "cmap_name")
    _sync(illum_cmap_sel, "value", "illum_cmap_name")

    def _on_illum_cmap_change(_):
        update_illum_image(state.illum_cmap_name)
        _illum_refs["cmap"] = state.illum_cmap_name
        _refresh_cbar()

    state.param.watch(_on_illum_cmap_change, "illum_cmap_name")

    def _section(label, *widgets):
        return pn.Column(
            pn.pane.Markdown(
                f"### {label}",
                styles={"color": "#93c5fd", "margin": "6px 0 2px 0"},
            ),
            *widgets,
        )

    def _subsection(label, *widgets):
        """Lighter sub-heading for grouped controls within a section."""
        return pn.Column(
            pn.pane.Markdown(
                f"**{label}**",
                styles={"color": "#7dd3fc", "font-size": "12px",
                        "margin": "6px 0 2px 0"},
            ),
            *widgets,
        )

    sidebar = pn.Column(
        pn.pane.Markdown(
            "### Artemis III Explorer",
            styles={"color": "#facc15", "font-size": "15px",
                    "margin": "0 0 2px 0"},
        ),
        pn.pane.Markdown(
            "_Public guide to the lunar south pole_",
            styles={"color": "#6b7280", "font-size": "11px",
                    "margin": "0 0 8px 0"},
        ),
        pn.layout.Divider(),

        _section(
            "3D Terrain View",
            _subsection(
                "Camera position",
                azimuth_sl,
                tilt_sl,
            ),
            _subsection(
                "Vertical relief",
                vexag_sl,
                pn.pane.Markdown(
                    "_Higher values make craters appear deeper. "
                    "2c is a natural, undistorted view._",
                    styles={"color": "#6b7280", "font-size": "10px"},
                ),
            ),
        ),
        pn.layout.Divider(),

        _section(
            "Colour Data",
            _subsection(
                "3D disc surface",
                cmap_sel,
                pn.pane.Markdown(
                    "_Colour scheme for elevation. batlow is perceptually "
                    "uniform and colour-blind safe [Crameri et al. 2020]._",
                    styles={"color": "#6b7280", "font-size": "10px"},
                ),
            ),
            _subsection(
                "2D illumination map",
                illum_cmap_sel,
                pn.pane.Markdown(
                    "_cividis: blue (shadowed) to yellow (sunlit)._\n\n"
                    "_plasma: purple to yellow, higher contrast._\n\n"
                    "_gray: scientific greyscale._",
                    styles={"color": "#6b7280", "font-size": "10px",
                            "line-height": "1.6"},
                ),
                slope_overlay_tog,
                pn.pane.HTML(
                    """
<div style="font-size:10px;color:#6b7280;line-height:1.8;margin-top:4px;">
  <span style="display:inline-block;width:12px;height:12px;
    border-radius:2px;background:#22c55e;opacity:0.22;
    vertical-align:middle;margin-right:5px;"></span>
  Slope ≤5&#xb0; — safe to land<br>
  <span style="display:inline-block;width:12px;height:12px;
    border-radius:2px;background:#fbbf24;opacity:0.82;
    vertical-align:middle;margin-right:5px;"></span>
  Slope 5–15&#xb0; — caution<br>
  <span style="display:inline-block;width:12px;height:12px;
    border-radius:2px;background:#ef4444;opacity:0.90;
    vertical-align:middle;margin-right:5px;"></span>
  Slope &gt;15&#xb0; — unsafe for landing
</div>
""",
                    width=220,
                ),
            ),
        ),
        pn.layout.Divider(),

        render_btn,
        pn.pane.Markdown(
            "_Press after changing any setting above._",
            styles={"color": "#6b7280", "font-size": "10px",
                    "margin": "2px 0 6px 0"},
        ),
        status_pane,
        pn.layout.Divider(),

        _section(
            "Guided Tour",
            tour_btn,
            pn.pane.Markdown(
                "_Cycles through Shackleton (ice), Connecting Ridge "
                "(light), and Malapert (comms)._",
                styles={"color": "#6b7280", "font-size": "10px",
                        "line-height": "1.6"},
            ),
        ),
        pn.layout.Divider(),
        pn.pane.HTML(
            """
<div style="font-size:11px;color:#94a3b8;line-height:2.0;">
  <strong style="color:#7dd3fc;display:block;margin-bottom:4px;">
    Site categories
  </strong>
  <span style="font-size:14px;">❄</span>&nbsp; Water ice access<br>
  <span style="font-size:14px;">☀</span>&nbsp; Solar power peak<br>
  <span style="font-size:14px;">⬡</span>&nbsp; Communications relay
</div>
""",
            width=220,
        ),
        pn.layout.Divider(),

        pn.pane.Markdown(
            "**Data sources**\n\n"
            "Elevation: LROC GLD100 [Scholten et al. 2012]\n\n"
            "Illumination: LROC colour poles composite\n\n"
            "Sites: NASA SDTT 2020 candidate regions",
            styles={"color": "#4b5563", "font-size": "10px",
                    "line-height": "1.7"},
        ),

        width=260,
        styles={
            "background": "#111827",
            "padding": "14px",
            "border-radius": "8px",
        },
    )

    main = pn.Column(
        pn.pane.Markdown(
            "## Artemis III — Understanding the Lunar South Pole",
            styles={"color": "#e2e8f0", "font-size": "22px",
                    "letter-spacing": "0.03em", "margin-bottom": "4px"},
        ),
        pn.pane.Markdown(
            "_An interactive guide for NASA's return to the Moon_",
            styles={"color": "#94a3b8", "font-size": "13px",
                    "margin-bottom": "12px"},
        ),

        intro_and,
        intro_but,
        intro_therefore,
        pn.layout.Divider(),

        pn.pane.Markdown(
            "### The Science: Why Shadows Never Leave",
            styles={"color": "#facc15", "font-size": "16px",
                    "margin": "4px 0 4px 0"},
        ),
        tilt_pane,
        pn.layout.Divider(),

        pn.pane.Markdown(
            "### 1 — The Terrain",
            styles={"color": "#facc15", "font-size": "16px",
                    "margin": "4px 0 2px 0"},
        ),
        pn.pane.Markdown(
            "The lunar south pole is defined by extreme rugged terrain. "
            "Deep craters plunge 4 km below their rims (with potential ice deposits). Ridges stand "
            "kilometres above the crater floors, perfect for solar power. Rotate the view to "
            "explore the landscape from any direction.",
            styles={"color": "#94a3b8", "font-size": "13px",
                    "line-height": "1.7", "margin-bottom": "8px"},
        ),
        pn.Column(
            pn.pane.Markdown(
                "**South Pole Terrain: 3D Polar View**  "
                "_(use the Rotate and Tilt sliders to orbit the terrain)_",
                styles={"color": "#d1d5db", "font-size": "13px"},
            ),
            disc_pane,
        ),
        pn.layout.Divider(),

        pn.pane.Markdown(
            "### 2 — The Light",
            styles={"color": "#facc15", "font-size": "16px",
                    "margin": "4px 0 2px 0"},
        ),
        pn.pane.Markdown(
            "Solar illumination at the south pole is interesting.Crater floors receive **no sunlight** for billions of "
            "years while nearby ridges sit in **near-constant sunlight**. "
            "Mission planners must find a site that balances both: "
            "enough light for solar power, close enough to shadow for "
            "water ice science. Click a gold star to explore a candidate site.",
            styles={"color": "#94a3b8", "font-size": "13px",
                    "line-height": "1.7", "margin-bottom": "8px"},
        ),
        pn.Column(
            pn.pane.Markdown(
                "**Solar Illumination Disc**  "
                "_(azimuthal equidistant, -90° to -83°  ·  "
                "yellow = sunlit  ·  purple = shadowed  ·  "
                "gold stars = candidate sites)_\n\n"
                "_Icons: ❄ = water ice access  ·  ☀ = solar power peak  "
                "·  ⬡ = communications relay  ·  click a star to learn more_",
                styles={"color": "#d1d5db", "font-size": "13px"},
            ),
            pn.Row(
                illum_col,
                _cbar_pane,
                align="start",
            ),
            pn.layout.Divider(),
            pn.pane.Markdown(
                "**Selected site:**",
                styles={"color": "#7dd3fc", "font-size": "13px",
                        "margin": "4px 0 2px 0"},
            ),
            info_pane,
            site_card_label,
            site_card_pane,
        ),
        pn.layout.Divider(),

        pn.pane.Markdown(
            "### 3 — Compare the Sites",
            styles={"color": "#facc15", "font-size": "16px",
                    "margin": "4px 0 2px 0"},
        ),
        pn.pane.Markdown(
            "Length encodes solar exposure more accurately than colour "
            "alone. The red dashed line marks 70% illumination, an arbitary "
            "estimated minimum for a solar-powered base to survive. "
            "Click a gold star on the map above "
            "to highlight that site on this chart.",
            styles={"color": "#94a3b8", "font-size": "13px",
                    "line-height": "1.7", "margin-bottom": "8px"},
        ),
        comparison_pane,

        styles={"background": "#0f172a", "padding": "16px"},
    )

    template = pn.template.FastListTemplate(
        title="Artemis III — Understanding the Lunar South Pole",
        sidebar=[sidebar],
        main=[main],
        theme="dark",
        accent="#facc15",
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

    print("[startup] Loading dataset ...", flush=True)
    DM = DataManager()

    print("[startup] Starting server at http://localhost:5007", flush=True)
    print("[startup] Press Ctrl+C to stop.", flush=True)

    pn.serve(
        create_app2,
        port=5007,
        show=True,
        title="Artemis III — Lunar South Pole",
        autoreload=False,
    )
