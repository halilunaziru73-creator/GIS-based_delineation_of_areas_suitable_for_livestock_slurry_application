import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from scipy.spatial.distance import pdist, squareform

from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
OUT = str(BASE / "outputs_data")
df = pd.read_csv(f"{OUT}/sample_points.csv")
test = df[df.split == "test"].reset_index(drop=True)
yte = test["SI_obs"].values
xy_te = test[["x", "y"]].values

preds = {
    "RK_LM": np.load(f"{OUT}/RK_LM_pred_test.npy"),
    "RK_RF": np.load(f"{OUT}/RK_RF_pred_test.npy"),
    "DL_MLP": np.load(f"{OUT}/DL_MLP_pred_test.npy"),
    "DL_CNN": np.load(f"{OUT}/DL_CNN_pred_test.npy"),
}

def morans_i(coords, values, k=8):
    d = squareform(pdist(coords))
    n = len(values)
    W = np.zeros((n, n))
    for i in range(n):
        nn = np.argsort(d[i])[1:k+1]
        W[i, nn] = 1.0
    W = W / W.sum()
    z = values - values.mean()
    num = np.sum(W * np.outer(z, z))
    den = np.sum(z**2) / n
    return float((n / W.sum() ) * 0 + (num / den))  # normalised below

def morans_i_correct(coords, values, k=8):
    d = squareform(pdist(coords))
    n = len(values)
    W = np.zeros((n, n))
    for i in range(n):
        nn = np.argsort(d[i])[1:k+1]
        W[i, nn] = 1.0
    S0 = W.sum()
    z = values - values.mean()
    num = n * np.sum(W * np.outer(z, z))
    den = S0 * np.sum(z**2)
    return float(num / den)

taylor = {}
moran = {}
for name, pred in preds.items():
    resid = yte - pred
    moran[name] = morans_i_correct(xy_te, resid, k=8)
    taylor[name] = dict(
        std=float(np.std(pred, ddof=1)),
        corr=float(np.corrcoef(pred, yte)[0, 1]),
        rmse_centered=float(np.sqrt(np.mean(((pred - pred.mean()) - (yte - yte.mean()))**2))),
    )

taylor["Reference"] = dict(std=float(np.std(yte, ddof=1)), corr=1.0, rmse_centered=0.0)

with open(f"{OUT}/validation_extra.json", "w") as f:
    json.dump({"morans_I_residuals": moran, "taylor_stats": taylor}, f, indent=2)
print("Moran's I of test residuals:", moran)
print("Taylor stats:", taylor)

# ---- permutation importance of the RK-RF trend model on covariates -------
FEATS = ["slope_pct", "dist_water_m", "dist_urban_m", "dist_protected_m", "nvz", "elevation"]
train = df[df.split == "train"]
rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
rf.fit(train[FEATS].values, train["SI_obs"].values)
imp = permutation_importance(rf, test[FEATS].values, yte, n_repeats=30, random_state=42)
importance = {f: dict(mean=float(m), std=float(s)) for f, m, s in
              zip(FEATS, imp.importances_mean, imp.importances_std)}
with open(f"{OUT}/feature_importance.json", "w") as f:
    json.dump(importance, f, indent=2)
print("Permutation importance:", importance)
