"""
Stage 3: simulate a field/expert-assessment sampling network. No physical
monitoring network exists in the supplied data, so - as agreed - a set of
pseudo-observation points is drawn from the candidate zone, the reference SI
surface (Stage 2) is sampled at each point, and heteroscedastic Gaussian noise
is added to emulate realistic field/laboratory assessment error. This is a
standard benchmarking design used to validate spatial-interpolation and
machine-learning methods in digital soil mapping (e.g. Hengl et al., 2004,
2007) when point observations are the object of study rather than the map
itself.
"""
import json
import numpy as np
import pandas as pd

from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
OUT = str(BASE / "outputs_data")
rng = np.random.default_rng(20260802)

cov = np.load(f"{OUT}/covariates.npz")
ref = np.load(f"{OUT}/mcda_reference.npz")
meta = json.load(open(f"{OUT}/grid_meta.json"))

transform = meta["transform"]  # [a,b,c,d,e,f] affine (rasterio order: a,b,c,d,e,f)
a, b, c, d, e, f = transform
def rc_to_xy(row, col):
    x = a*col + b*row + c
    y = d*col + e*row + f
    return x, y

candidate_mask = ref["candidate_mask"]
SI = ref["SI"]
slope_pct = cov["slope_pct"]
dist_water = cov["dist_water"]
dist_urban = cov["dist_urban"]
dist_protected = cov["dist_protected"]
nvz_mask = cov["nvz_mask"]
dem = cov["dem"]

rows, cols = np.where(candidate_mask)
n_candidates = len(rows)
N_SAMPLES = 550
idx = rng.choice(n_candidates, size=N_SAMPLES, replace=False)
sample_rows, sample_cols = rows[idx], cols[idx]

records = []
for r, cix in zip(sample_rows, sample_cols):
    x, y = rc_to_xy(r, cix)
    si_true = SI[r, cix]
    noise_sd = 3.0 + 0.06*si_true          # heteroscedastic assessment error
    si_obs = float(si_true + rng.normal(0, noise_sd))
    si_obs = float(np.clip(si_obs, 0, 100))
    records.append(dict(
        sample_id=len(records)+1, row=int(r), col=int(cix), x=float(x), y=float(y),
        elevation=float(dem[r, cix]), slope_pct=float(slope_pct[r, cix]),
        dist_water_m=float(dist_water[r, cix]), dist_urban_m=float(dist_urban[r, cix]),
        dist_protected_m=float(dist_protected[r, cix]), nvz=int(nvz_mask[r, cix]),
        SI_true=float(si_true), SI_obs=si_obs,
    ))

df = pd.DataFrame(records)

# ---- spatial block split (5x5 km blocks) for train/test to avoid leakage --
block = 2000.0  # metres
df["block_x"] = (df.x // block).astype(int)
df["block_y"] = (df.y // block).astype(int)
df["block_id"] = df["block_x"].astype(str) + "_" + df["block_y"].astype(str)
blocks = np.array(sorted(df["block_id"].unique()), dtype=object)
perm = rng.permutation(len(blocks))
blocks = blocks[perm]
n_test_blocks = max(1, int(round(0.25*len(blocks))))
test_blocks = set(blocks[:n_test_blocks])
df["split"] = np.where(df["block_id"].isin(test_blocks), "test", "train")

print(df["split"].value_counts())
print(df[["SI_true", "SI_obs"]].describe())

df.to_csv(f"{OUT}/sample_points.csv", index=False)
print(f"Saved {len(df)} pseudo-observation points -> sample_points.csv "
      f"({(df.split=='train').sum()} train / {(df.split=='test').sum()} test, "
      f"{len(blocks)} spatial blocks, {n_test_blocks} held out)")
