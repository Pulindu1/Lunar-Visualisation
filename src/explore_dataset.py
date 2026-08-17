"""
Dataset exploration -- inventories the lunar rasters and checks their value
ranges, distributions and correlations, to inform the visualisation design.

Run from the project root (paths below are relative to it):
    python3 src/explore_dataset.py

Writes plots to explore_graphs/.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HEIGHTMAP_DIR = "dataset/heightmaps"
ILLUMINATION_DIR = "dataset/illumination"
OUTPUT_DIR = "explore_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1a. Inventory
# ---------------------------------------------------------------------------
VALID_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
hmaps = sorted(f for f in os.listdir(HEIGHTMAP_DIR) if os.path.splitext(f)[1].lower() in VALID_EXT)
illums = sorted(f for f in os.listdir(ILLUMINATION_DIR) if os.path.splitext(f)[1].lower() in VALID_EXT)
print("=" * 60)
print(f"Heightmaps  ({len(hmaps)}): {hmaps}")
print(f"Illuminations ({len(illums)}): {illums}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1a. Image format, bit depth, resolution, value range
# ---------------------------------------------------------------------------
print("\n--- HEIGHTMAPS ---")
hmap_arrays = {}
for fname in hmaps:
    path = os.path.join(HEIGHTMAP_DIR, fname)
    img = Image.open(path)
    arr = np.array(img)
    hmap_arrays[fname] = arr
    print(f"\n{fname}")
    print(f"  mode={img.mode}  size={img.size}  format={img.format}")
    print(f"  dtype={arr.dtype}  shape={arr.shape}")
    print(f"  min={arr.min()}  max={arr.max()}  mean={arr.mean():.2f}  std={arr.std():.2f}")
    # If 16-bit uint, report elevation in metres using LROC scale (0.5 m/DN)
    if arr.dtype in [np.uint16, np.int16] or img.mode in ("I;16", "I"):
        elev_min = arr.min() * 0.5
        elev_max = arr.max() * 0.5
        print(f"  >> Estimated elevation range (0.5 m/DN): {elev_min:.0f} m  to  {elev_max:.0f} m")

print("\n--- ILLUMINATION ---")
illum_arrays = {}
for fname in illums:
    path = os.path.join(ILLUMINATION_DIR, fname)
    img = Image.open(path)
    arr = np.array(img)
    illum_arrays[fname] = arr
    print(f"\n{fname}")
    print(f"  mode={img.mode}  size={img.size}  format={img.format}")
    print(f"  dtype={arr.dtype}  shape={arr.shape}")
    print(f"  min={arr.min()}  max={arr.max()}  mean={arr.mean():.2f}")
    if arr.ndim == 3:
        print(f"  channels={arr.shape[2]}  (colour image)")

# ---------------------------------------------------------------------------
# 1b. Histogram per heightmap + visual preview
# ---------------------------------------------------------------------------
n = len(hmaps)
fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
if n == 1:
    axes = axes.reshape(2, 1)

for i, fname in enumerate(hmaps):
    arr = hmap_arrays[fname].astype(np.float32)
    # Squeeze to 2-D if needed (e.g. mode=RGB stored as heightmap)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    axes[0, i].imshow(arr, cmap="gray")
    axes[0, i].set_title(fname, fontsize=8)
    axes[0, i].axis("off")
    axes[1, i].hist(arr.flatten(), bins=256, color="steelblue", log=True)
    axes[1, i].set_xlabel("DN value")
    axes[1, i].set_ylabel("Count (log)")

axes[0, 0].set_ylabel("Heightmap")
plt.suptitle("Heightmap previews + histograms", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "explore_heightmaps.png"), dpi=150)
print("\nSaved: explore_graphs/explore_heightmaps.png")
plt.show()

# ---------------------------------------------------------------------------
# Illumination previews
# ---------------------------------------------------------------------------
n_il = len(illums)
fig2, axes2 = plt.subplots(2, n_il, figsize=(5 * n_il, 8))
if n_il == 1:
    axes2 = axes2.reshape(2, 1)

for i, fname in enumerate(illums):
    arr = illum_arrays[fname]
    # Use luminance for colour images
    if arr.ndim == 3:
        lum = arr.mean(axis=2).astype(np.float32)
    else:
        lum = arr.astype(np.float32)
    axes2[0, i].imshow(arr)
    axes2[0, i].set_title(fname, fontsize=8)
    axes2[0, i].axis("off")
    axes2[1, i].hist(lum.flatten(), bins=256, color="darkorange", log=True)
    axes2[1, i].set_xlabel("Pixel value (luminance)")
    axes2[1, i].set_ylabel("Count (log)")

plt.suptitle("Illumination previews + histograms", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "explore_illuminations.png"), dpi=150)
print("Saved: explore_graphs/explore_illuminations.png")
plt.show()

# ---------------------------------------------------------------------------
# 1c. Spatial correspondence check
# ---------------------------------------------------------------------------
print("\n--- SPATIAL CORRESPONDENCE ---")
# Use the highest-resolution heightmap and illumination for comparison
ref_h_name = hmaps[-1]   # last = likely highest-res after sort
ref_il_name = illums[-1]

h_ref = hmap_arrays[ref_h_name].astype(np.float32)
il_ref = illum_arrays[ref_il_name].astype(np.float32)
if h_ref.ndim == 3:
    h_ref = h_ref.mean(axis=2)
if il_ref.ndim == 3:
    il_ref = il_ref.mean(axis=2)

print(f"Reference heightmap : {ref_h_name}  shape={h_ref.shape}")
print(f"Reference illumin.  : {ref_il_name}  shape={il_ref.shape}")
if h_ref.shape == il_ref.shape:
    print("  >> Shapes match -- 1:1 pixel correspondence.")
else:
    print("  >> Shapes differ -- will need resampling for overlay.")

# Side-by-side overlay at common scale
fig3, (ax_h, ax_il) = plt.subplots(1, 2, figsize=(12, 5))
ax_h.imshow(h_ref, cmap="gray")
ax_h.set_title(f"Heightmap: {ref_h_name}")
ax_il.imshow(il_ref, cmap="gray")
ax_il.set_title(f"Illumination: {ref_il_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "explore_overlay.png"), dpi=150)
print("Saved: explore_graphs/explore_overlay.png")
plt.show()

# ---------------------------------------------------------------------------
# 1d. Elevation vs illumination correlation
# ---------------------------------------------------------------------------
print("\n--- CORRELATION: ELEVATION vs ILLUMINATION ---")

# Resample illumination to heightmap size if needed
from PIL import Image as PILImage

h_img = PILImage.fromarray(h_ref.astype(np.float32))
il_pil = PILImage.open(os.path.join(ILLUMINATION_DIR, ref_il_name)).convert("L")
il_pil_r = il_pil.resize((h_ref.shape[1], h_ref.shape[0]), PILImage.LANCZOS)
il_matched = np.array(il_pil_r).astype(np.float32)

h_norm = (h_ref - h_ref.min()) / (h_ref.max() - h_ref.min() + 1e-9)
il_norm = il_matched / (il_matched.max() + 1e-9)

# Subsample for speed
step = max(1, h_norm.size // 50000)
h_flat = h_norm.flatten()[::step]
il_flat = il_norm.flatten()[::step]

# Bin the scatter to a 2-D density map for clarity
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
axes4[0].scatter(h_flat, il_flat, alpha=0.05, s=1, color="steelblue")
axes4[0].set_xlabel("Normalised elevation")
axes4[0].set_ylabel("Normalised illumination")
axes4[0].set_title("Elevation vs Illumination (scatter)")

h2d, xe, ye = np.histogram2d(h_flat, il_flat, bins=100)
axes4[1].imshow(
    np.log1p(h2d.T),
    origin="lower",
    extent=[xe[0], xe[-1], ye[0], ye[-1]],
    aspect="auto",
    cmap="inferno",
)
axes4[1].set_xlabel("Normalised elevation")
axes4[1].set_ylabel("Normalised illumination")
axes4[1].set_title("Elevation vs Illumination (log-density)")

corr = np.corrcoef(h_flat, il_flat)[0, 1]
print(f"  Pearson r (elevation, illumination) = {corr:.4f}")

plt.suptitle(f"Elevation–Illumination correlation  (r={corr:.3f})", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "explore_correlation.png"), dpi=150)
print("Saved: explore_graphs/explore_correlation.png")
plt.show()

# ---------------------------------------------------------------------------
# 1e. Permanently Shadowed Regions (PSRs)
# ---------------------------------------------------------------------------
print("\n--- PERMANENTLY SHADOWED REGIONS ---")

# The illumination images are false-colour RGB -- pixel values don't go to 0.
# PSRs are the darkest regions, so use the 10th percentile of luminance as the
# threshold rather than a fixed fraction of the maximum.
il_colour = np.array(
    Image.open(os.path.join(ILLUMINATION_DIR, ref_il_name))
    .resize((h_ref.shape[1], h_ref.shape[0]), Image.LANCZOS)
)
il_lum = il_colour.mean(axis=2).astype(np.float32)  # perceptual luminance proxy

psr_thresh = float(np.percentile(il_lum, 10))        # darkest 10% = PSR candidates
psr_mask = il_lum < psr_thresh
psr_pct = psr_mask.sum() / psr_mask.size * 100

print(f"  Luminance range : {il_lum.min():.1f} – {il_lum.max():.1f}")
print(f"  PSR threshold (10th pct): {psr_thresh:.1f}")
print(f"  PSR coverage : {psr_pct:.1f}% of image area")

fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5))
axes5[0].imshow(il_colour)
axes5[0].set_title(f"Illumination colour: {ref_il_name}")
axes5[0].axis("off")
axes5[1].imshow(il_lum, cmap="gray")
axes5[1].set_title("Luminance channel")
axes5[1].axis("off")
axes5[2].imshow(psr_mask, cmap="Blues")
axes5[2].set_title(f"PSR mask (darkest 10%)  [{psr_pct:.1f}% of area]")
axes5[2].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "explore_psr.png"), dpi=150)
print("Saved: explore_graphs/explore_psr.png")
plt.show()

# ---------------------------------------------------------------------------
# 1f. Slope map (landing safety proxy)
# ---------------------------------------------------------------------------
print("\n--- SLOPE MAP ---")

gy, gx = np.gradient(h_ref.astype(np.float64))
slope = np.sqrt(gx**2 + gy**2)
slope_mean = slope.mean()
slope_p95 = np.percentile(slope, 95)
print(f"  Slope  mean={slope_mean:.3f}  95th pct={slope_p95:.3f} DN/pixel")

fig6, axes6 = plt.subplots(1, 2, figsize=(12, 5))
im = axes6[0].imshow(slope, cmap="hot", vmax=slope_p95)
fig6.colorbar(im, ax=axes6[0], label="Slope (DN/pixel)")
axes6[0].set_title("Slope map (landing safety proxy)")
axes6[1].hist(slope.flatten(), bins=256, color="firebrick", log=True)
axes6[1].axvline(slope_p95, color="black", linestyle="--", label="95th pct")
axes6[1].set_xlabel("Slope (DN/pixel)")
axes6[1].set_ylabel("Count (log)")
axes6[1].set_title("Slope distribution")
axes6[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "explore_slope.png"), dpi=150)
print("Saved: explore_graphs/explore_slope.png")
plt.show()

# ---------------------------------------------------------------------------
# Summary for design decisions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("EXPLORATION SUMMARY")
print("=" * 60)
for fname in hmaps:
    arr = hmap_arrays[fname]
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    print(f"\nHeightmap: {fname}")
    print(f"  shape={arr.shape}  dtype={arr.dtype}")
    print(f"  DN range: [{arr.min()}, {arr.max()}]")
    if arr.dtype in [np.uint16, np.int16]:
        print(f"  Elevation range (0.5m/DN): [{arr.min()*0.5:.0f}m, {arr.max()*0.5:.0f}m]")
print(f"\nPSR coverage: {psr_pct:.1f}%")
print(f"Elevation–Illumination correlation: r={corr:.4f}")
print(f"Slope 95th percentile: {slope_p95:.3f} DN/pixel")
print("\nKey design decisions:")
print("  - Use 16-bit heightmaps (higher DN precision) for main vis")
print("  - Colourmap: perceptually uniform (e.g. 'plasma' or custom LROC palette)")
print("  - Overlay PSR mask in blue-tint on 3D view")
print("  - Click-to-sample: report elevation in metres (DN * 0.5)")
print("  - Contour interval: ~200 m (40 DN for 16-bit images)")
print("=" * 60)
