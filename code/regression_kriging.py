"""
Stage 4: Regression Kriging (RK).
  RK = deterministic trend (regression on covariates) + kriged interpolation
       of the trend's residuals (Odeh et al., 1995; Hengl et al., 2004, 2007).
Two trend models are compared:
  - RK-LM : ordinary least squares trend (the classical RK formulation)
  - RK-RF : random-forest trend (a modern, non-linear extension)
Both use ordinary kriging of residuals with a spherical variogram model.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pykrige.rk import RegressionKriging
from pykrige.ok import OrdinaryKriging

OUT = "/home/claude/work/out"
cov = np.load(f"{OUT}/covariates.npz")
ref = np.load(f"{OUT}/mcda_reference.npz")
meta = json.load(open(f"{OUT}/grid_meta.json"))
df = pd.read_csv(f"{OUT}/sample_points.csv")

FEATS = ["slope_pct", "dist_water_m", "dist_urban_m", "dist_protected_m", "nvz", "elevation"]

train = df[df.split == "train"].reset_index(drop=True)
test  = df[df.split == "test"].reset_index(drop=True)

Xtr, ytr = train[FEATS].values, train["SI_obs"].values
Xte, yte = test[FEATS].values,  test["SI_obs"].values
xy_tr = train[["x", "y"]].values
xy_te = test[["x", "y"]].values

results = {}
models = {}

for name, reg in [
    ("RK_LM", LinearRegression()),
    ("RK_RF", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)),
]:
    m = RegressionKriging(regression_model=reg, method="ordinary",
                           variogram_model="spherical", n_closest_points=12,
                           weight=True, exact_values=True)
    m.fit(Xtr, xy_tr, ytr)
    pred_test = m.predict(Xte, xy_te)
    rmse = float(np.sqrt(mean_squared_error(yte, pred_test)))
    mae  = float(mean_absolute_error(yte, pred_test))
    r2   = float(r2_score(yte, pred_test))
    results[name] = dict(rmse=rmse, mae=mae, r2=r2)
    models[name] = m
    print(f"{name}: test RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    np.save(f"{OUT}/{name}_pred_test.npy", pred_test)

# ---- variogram of OLS-trend residuals (for the figure) --------------------
lm = LinearRegression().fit(Xtr, ytr)
resid = ytr - lm.predict(Xtr)
ok = OrdinaryKriging(xy_tr[:, 0], xy_tr[:, 1], resid,
                      variogram_model="spherical", nlags=12, verbose=False,
                      enable_plotting=False)
lags, gamma = ok.lags, ok.semivariance
vparams = ok.variogram_model_parameters  # [sill-nugget(partial sill), range, nugget]

np.savez_compressed(f"{OUT}/variogram_residuals.npz",
                     lags=lags, gamma=gamma, params=np.array(vparams),
                     resid=resid, xy=xy_tr)

# ---- full-grid prediction (within candidate zone) for the best model ------
candidate_mask = ref["candidate_mask"]
rows, cols = np.where(candidate_mask)
a, b, c, d, e, f = meta["transform"]
gx = a*cols + b*rows + c
gy = d*cols + e*rows + f

grid_feats = np.column_stack([
    cov["slope_pct"][rows, cols], cov["dist_water"][rows, cols],
    cov["dist_urban"][rows, cols], cov["dist_protected"][rows, cols],
    cov["nvz_mask"][rows, cols].astype(float), cov["dem"][rows, cols],
])
xy_grid = np.column_stack([gx, gy])

best_name = min(results, key=lambda k: results[k]["rmse"])
print("Best RK model on held-out spatial blocks:", best_name)

best_model = models[best_name]
grid_pred_full = best_model.predict(grid_feats, xy_grid)

# kriging variance surface (from the Krige wrapper used inside RegressionKriging)
pts = best_model.krige._dimensionality_check(xy_grid, ext="points")
_, ok_var = best_model.krige.execute(pts)

SI_map = np.full(cov["dem"].shape, np.nan, dtype=np.float32)
VAR_map = np.full(cov["dem"].shape, np.nan, dtype=np.float32)
SI_map[rows, cols] = grid_pred_full
VAR_map[rows, cols] = ok_var

np.savez_compressed(f"{OUT}/RK_grid_prediction.npz",
                     SI_map=SI_map, VAR_map=VAR_map, best_name=best_name)

with open(f"{OUT}/rk_results.json", "w") as fjs:
    json.dump({"results": results, "best_model": best_name,
               "variogram_params_sill_range_nugget": list(map(float, vparams))}, fjs, indent=2)

print("Saved RK_grid_prediction.npz, variogram_residuals.npz, rk_results.json")
