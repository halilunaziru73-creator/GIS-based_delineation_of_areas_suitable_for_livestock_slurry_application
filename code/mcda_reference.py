"""
Stage 2: (a) reproduce the rule-based binary MCDA overlay (hard constraints,
following the original Tudela report's 7 criteria) and (b) build a continuous,
weighted-linear-combination fuzzy Suitability Index (SI, 0-100) *within* the
hard-constraint candidate zone. SI is the reference/"latent" surface that the
pseudo-observation network (Stage 3) samples, and that regression kriging and
the deep-learning models (Stages 4-5) attempt to reconstruct from covariates.
"""
import json
import numpy as np

from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
OUT = str(BASE / "outputs_data")
z = np.load(f"{OUT}/covariates.npz")

slope_pct       = z["slope_pct"]
soil_class01    = z["soil_class01"].astype(bool)
dist_water      = z["dist_water"]
water_mask      = z["water_mask"].astype(bool)
dist_urban      = z["dist_urban"]
urban_buffer    = z["urban_buffer_mask"].astype(bool)
dist_protected  = z["dist_protected"]
pna_mask        = z["pna_mask"].astype(bool)
nvz_mask        = z["nvz_mask"].astype(bool)
land_eligible   = z["land_eligible"].astype(bool)
tudela_mask     = z["tudela_mask"].astype(bool)
aoi_mask        = z["aoi_mask"].astype(bool)
dem             = z["dem"]

# ---------------------------------------------------------------------
# Tier 1 - hard (categorical / regulatory) constraints -> candidate zone
# ---------------------------------------------------------------------
slope_excluded = slope_pct >= 15.0
hard_exclusion = (
    slope_excluded | (~soil_class01) | water_mask | urban_buffer |
    pna_mask | (~land_eligible)
)
candidate_mask = aoi_mask & (~hard_exclusion)

print(f"Candidate ('apt') zone: {100*candidate_mask[aoi_mask].mean():.2f}% of AOI, "
      f"{100*candidate_mask[tudela_mask].mean():.2f}% of Tudela municipality, "
      f"{candidate_mask.sum()} cells ({candidate_mask.sum()*25*25/1e6:.2f} km2)")

# ---------------------------------------------------------------------
# Tier 2 - continuous fuzzy Suitability Index within the candidate zone
# ---------------------------------------------------------------------
f_slope      = np.clip(1.0 - slope_pct / 15.0, 0.0, 1.0)
f_water      = np.clip(dist_water / 300.0, 0.0, 1.0)
f_urban      = np.clip((dist_urban - 100.0) / 400.0, 0.0, 1.0)
f_protected  = np.clip(dist_protected / 1000.0, 0.0, 1.0)
nvz_factor   = np.where(nvz_mask, 0.70, 1.0)   # agronomic N-load penalty, not exclusion

W = dict(slope=0.30, water=0.25, urban=0.20, protected=0.25)
SI_raw = (W["slope"]*f_slope + W["water"]*f_water +
          W["urban"]*f_urban + W["protected"]*f_protected)
SI = 100.0 * nvz_factor * SI_raw
SI = np.where(candidate_mask, SI, np.nan).astype(np.float32)

valid = candidate_mask
print(f"SI within candidate zone: mean={np.nanmean(SI):.1f}, sd={np.nanstd(SI):.1f}, "
      f"min={np.nanmin(SI):.1f}, max={np.nanmax(SI):.1f}")

np.savez_compressed(f"{OUT}/mcda_reference.npz",
    candidate_mask=candidate_mask, hard_exclusion=hard_exclusion,
    f_slope=f_slope, f_water=f_water, f_urban=f_urban, f_protected=f_protected,
    nvz_factor=nvz_factor, SI=SI, slope_excluded=slope_excluded)

stats = {
    "candidate_pct_of_AOI": float(100*candidate_mask[aoi_mask].mean()),
    "candidate_pct_of_Tudela": float(100*candidate_mask[tudela_mask].mean()),
    "candidate_area_km2": float(candidate_mask.sum()*25*25/1e6),
    "SI_mean": float(np.nanmean(SI)), "SI_sd": float(np.nanstd(SI)),
    "SI_min": float(np.nanmin(SI)), "SI_max": float(np.nanmax(SI)),
    "weights": W,
}
with open(f"{OUT}/mcda_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("Saved mcda_reference.npz + mcda_stats.json")
