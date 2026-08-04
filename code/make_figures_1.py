import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cartohelpers import add_scalebar, add_north_arrow, style_map_axes, credit_text

BASE = Path(__file__).resolve().parent.parent
OUT = str(BASE / "outputs_data")
FIGS = f"{OUT}/figs"
import os; os.makedirs(FIGS, exist_ok=True)

cov = np.load(f"{OUT}/covariates.npz")
ref = np.load(f"{OUT}/mcda_reference.npz")
meta = json.load(open(f"{OUT}/grid_meta.json"))
bounds = meta["bounds"]  # left,bottom,right,top
extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

dem = cov["dem"]; aoi_mask = cov["aoi_mask"]; tudela_mask = cov["tudela_mask"]
dem_disp = np.where(aoi_mask, dem, np.nan)

# hillshade
def hillshade(z, az=315, alt=45, dx=25, dy=25):
    az = np.radians(360.0 - az); alt = np.radians(alt)
    dzdy, dzdx = np.gradient(np.nan_to_num(z, nan=np.nanmean(z)), dy, dx)
    slope = np.pi/2 - np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, dzdy)
    shaded = np.sin(alt)*np.sin(slope) + np.cos(alt)*np.cos(slope)*np.cos(az - aspect)
    return np.clip(shaded, 0, 1)

hs = hillshade(dem_disp)

# ============================================================ FIGURE 1 =====
fig, ax = plt.subplots(figsize=(8, 6.2), dpi=300)
ax.imshow(hs, cmap="gray", extent=extent, origin="upper", alpha=0.55)
im = ax.imshow(dem_disp, cmap="terrain", extent=extent, origin="upper", alpha=0.75)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Elevation (m a.s.l.)", fontsize=8)
# municipal boundary
from rasterio.features import shapes as rio_shapes
import rasterio
for geom, val in rio_shapes(tudela_mask.astype(np.uint8), mask=tudela_mask,
                            transform=rasterio.transform.from_bounds(*bounds, dem.shape[1], dem.shape[0])):
    xs, ys = zip(*geom["coordinates"][0])
    ax.plot(xs, ys, color="black", lw=1.6)
style_map_axes(ax, "(a) Digital elevation model and municipal boundary \u2013 Tudela (Navarre, Spain)")
add_scalebar(ax, bounds, 3)
add_north_arrow(ax, bounds)
credit_text(fig, "Data: CNIG (DEM), IGN INSPIRE administrative boundaries. CRS: ETRS89 / UTM zone 30N (EPSG:25830).")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig1_study_area_dem.png", bbox_inches="tight")
plt.close(fig)
print("fig1 done")

# ============================================================ FIGURE 2 =====
# Hard-constraint criteria panel (2x3)
slope_excl = ref["slope_excluded"]
soil01 = cov["soil_class01"].astype(bool)
water = cov["water_mask"].astype(bool)
urban = cov["urban_buffer_mask"].astype(bool)
pna = cov["pna_mask"].astype(bool)
nvz = cov["nvz_mask"].astype(bool)
land = cov["land_eligible"].astype(bool)

panels = [
    ("Slope \u2265 15% (excluded)", np.where(aoi_mask, slope_excl, np.nan), "Reds"),
    ("Soil/lithology unsuitable", np.where(aoi_mask, ~soil01, np.nan), "Oranges"),
    ("Water-protection buffer (DPH + 5 m)", np.where(aoi_mask, water, np.nan), "Blues"),
    ("Urban 100 m buffer", np.where(aoi_mask, urban, np.nan), "Greys"),
    ("Protected natural areas", np.where(aoi_mask, pna, np.nan), "Greens"),
    ("Nitrate Vulnerable Zone", np.where(aoi_mask, nvz, np.nan), "Purples"),
]
fig, axs = plt.subplots(2, 3, figsize=(11, 7), dpi=300)
for a, (title, arr, cmap) in zip(axs.ravel(), panels):
    a.imshow(hs, cmap="gray", extent=extent, origin="upper", alpha=0.35)
    a.imshow(arr, cmap=ListedColormap(["none", plt.get_cmap(cmap)(0.75)]), extent=extent,
             origin="upper", vmin=0, vmax=1)
    style_map_axes(a, title)
fig.suptitle("Figure 2. Individual regulatory, environmental and physical exclusion criteria", fontsize=11, y=0.995)
credit_text(fig, "Shaded = area affected by each criterion, over hillshaded relief. CRS EPSG:25830.")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{FIGS}/fig2_exclusion_criteria.png", bbox_inches="tight")
plt.close(fig)
print("fig2 done")

# ============================================================ FIGURE 3 =====
candidate_mask = ref["candidate_mask"]
df = pd.read_csv(f"{OUT}/sample_points.csv")
fig, ax = plt.subplots(figsize=(8, 6.2), dpi=300)
ax.imshow(hs, cmap="gray", extent=extent, origin="upper", alpha=0.45)
cand_disp = np.where(aoi_mask, candidate_mask, np.nan)
ax.imshow(cand_disp, cmap=ListedColormap(["none", "#2ca25f"]), extent=extent, origin="upper", vmin=0, vmax=1)
for split, marker, color, lbl in [("train", "o", "#1f78b4", "Training points (n=%d)"),
                                    ("test", "^", "#e31a1c", "Held-out test points (n=%d)")]:
    sub = df[df.split == split]
    ax.scatter(sub.x, sub.y, s=14, marker=marker, color=color, edgecolor="black",
               linewidth=0.3, label=lbl % len(sub), zorder=5)
ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
style_map_axes(ax, "(a) Candidate zone (passes all hard constraints) and pseudo-observation network")
add_scalebar(ax, bounds, 3); add_north_arrow(ax, bounds)
credit_text(fig, "Candidate zone = slope<15% AND soil-suitable AND arable/pasture land use AND outside water/urban/protected buffers.")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig3_candidate_zone_samples.png", bbox_inches="tight")
plt.close(fig)
print("fig3 done")

# ============================================================ FIGURE 4 =====
SI = ref["SI"]
fig, ax = plt.subplots(figsize=(8, 6.2), dpi=300)
ax.imshow(hs, cmap="gray", extent=extent, origin="upper", alpha=0.35)
im = ax.imshow(SI, cmap="RdYlGn", extent=extent, origin="upper", vmin=0, vmax=100)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Suitability Index (SI, 0\u2013100)", fontsize=8)
style_map_axes(ax, "(a) Reference continuous fuzzy Suitability Index within the candidate zone")
add_scalebar(ax, bounds, 3); add_north_arrow(ax, bounds)
credit_text(fig, "SI = 100 x NVZ-penalty x weighted linear combination of fuzzy slope/water/urban/protected-distance memberships.")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig4_reference_SI.png", bbox_inches="tight")
plt.close(fig)
print("fig4 done")
