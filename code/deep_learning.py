"""
Stage 5: Deep-learning surrogates for the suitability index.
  - DL-MLP : dense feed-forward network on point covariates (+ coordinates)
  - DL-CNN : small 2-D convolutional network reading a 9x9 multichannel patch
             of covariate rasters centred on each location (captures local
             spatial context / texture that point-wise covariates cannot).
Both are trained on the pseudo-observation network (Stage 3) with an
80/20 train/validation split drawn from the training spatial blocks, and
evaluated on the held-out spatial-block test set (same test set as RK).
"""
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.random.set_seed(20260802)
np.random.seed(20260802)

OUT = "/home/claude/work/out"
cov = np.load(f"{OUT}/covariates.npz")
ref = np.load(f"{OUT}/mcda_reference.npz")
meta = json.load(open(f"{OUT}/grid_meta.json"))
df = pd.read_csv(f"{OUT}/sample_points.csv")

FEATS = ["slope_pct", "dist_water_m", "dist_urban_m", "dist_protected_m", "nvz", "elevation", "x", "y"]
train_full = df[df.split == "train"].reset_index(drop=True)
test = df[df.split == "test"].reset_index(drop=True)

# further split train -> train/val (random, within training blocks only)
rng = np.random.default_rng(1)
val_idx = rng.choice(len(train_full), size=int(0.2*len(train_full)), replace=False)
val_mask = np.zeros(len(train_full), dtype=bool); val_mask[val_idx] = True
train = train_full[~val_mask].reset_index(drop=True)
val   = train_full[val_mask].reset_index(drop=True)

scaler = StandardScaler().fit(train[FEATS].values)
Xtr = scaler.transform(train[FEATS].values); ytr = train["SI_obs"].values
Xva = scaler.transform(val[FEATS].values);   yva = val["SI_obs"].values
Xte = scaler.transform(test[FEATS].values);  yte = test["SI_obs"].values

# ---------------------------------------------------------------- DL-MLP --
def build_mlp(n_in):
    inp = keras.Input(shape=(n_in,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m

mlp = build_mlp(len(FEATS))
es = keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True, monitor="val_loss")
hist_mlp = mlp.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=400, batch_size=16,
                    verbose=0, callbacks=[es])
pred_mlp_test = mlp.predict(Xte, verbose=0).ravel()
mlp_rmse = float(np.sqrt(mean_squared_error(yte, pred_mlp_test)))
mlp_mae  = float(mean_absolute_error(yte, pred_mlp_test))
mlp_r2   = float(r2_score(yte, pred_mlp_test))
print(f"DL-MLP: test RMSE={mlp_rmse:.2f} MAE={mlp_mae:.2f} R2={mlp_r2:.3f} "
      f"(stopped at epoch {len(hist_mlp.history['loss'])})")

# ------------------------------------------------------------- DL-CNN ----
PATCH = 9
HALF = PATCH // 2
channels = np.stack([
    np.nan_to_num(cov["slope_pct"], nan=0.0),
    np.nan_to_num(cov["dist_water"], nan=0.0),
    np.nan_to_num(cov["dist_urban"], nan=0.0),
    np.nan_to_num(cov["dist_protected"], nan=0.0),
    cov["nvz_mask"].astype(np.float32),
    np.nan_to_num(cov["dem"], nan=np.nanmean(cov["dem"])),
], axis=-1).astype(np.float32)
H, W, C = channels.shape
pad = np.pad(channels, ((HALF, HALF), (HALF, HALF), (0, 0)), mode="edge")

ch_mean = channels.reshape(-1, C).mean(axis=0)
ch_std  = channels.reshape(-1, C).std(axis=0) + 1e-6

def extract_patches(rows, cols):
    out = np.empty((len(rows), PATCH, PATCH, C), dtype=np.float32)
    for i, (r, cix) in enumerate(zip(rows, cols)):
        out[i] = pad[r:r+PATCH, cix:cix+PATCH, :]
    return (out - ch_mean) / ch_std

Ptr = extract_patches(train.row.values, train.col.values)
Pva = extract_patches(val.row.values, val.col.values)
Pte = extract_patches(test.row.values, test.col.values)

def build_cnn(patch, c):
    inp = keras.Input(shape=(patch, patch, c))
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inp)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="linear")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m

cnn = build_cnn(PATCH, C)
es2 = keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True, monitor="val_loss")
hist_cnn = cnn.fit(Ptr, ytr, validation_data=(Pva, yva), epochs=400, batch_size=16,
                    verbose=0, callbacks=[es2])
pred_cnn_test = cnn.predict(Pte, verbose=0).ravel()
cnn_rmse = float(np.sqrt(mean_squared_error(yte, pred_cnn_test)))
cnn_mae  = float(mean_absolute_error(yte, pred_cnn_test))
cnn_r2   = float(r2_score(yte, pred_cnn_test))
print(f"DL-CNN: test RMSE={cnn_rmse:.2f} MAE={cnn_mae:.2f} R2={cnn_r2:.3f} "
      f"(stopped at epoch {len(hist_cnn.history['loss'])})")

# ---- full-grid prediction within candidate zone ---------------------------
candidate_mask = ref["candidate_mask"]
rows, cols = np.where(candidate_mask)
a, b, c_, d, e, f = meta["transform"]
gx = a*cols + b*rows + c_
gy = d*cols + e*rows + f

grid_X = np.column_stack([
    cov["slope_pct"][rows, cols], cov["dist_water"][rows, cols],
    cov["dist_urban"][rows, cols], cov["dist_protected"][rows, cols],
    cov["nvz_mask"][rows, cols].astype(float), cov["dem"][rows, cols], gx, gy,
])
grid_X_scaled = scaler.transform(grid_X)
mlp_grid_pred = mlp.predict(grid_X_scaled, verbose=0).ravel()

grid_patches = extract_patches(rows, cols)
cnn_grid_pred = cnn.predict(grid_patches, batch_size=512, verbose=0).ravel()

MLP_map = np.full(cov["dem"].shape, np.nan, dtype=np.float32)
CNN_map = np.full(cov["dem"].shape, np.nan, dtype=np.float32)
MLP_map[rows, cols] = mlp_grid_pred
CNN_map[rows, cols] = cnn_grid_pred

np.savez_compressed(f"{OUT}/DL_grid_predictions.npz", MLP_map=MLP_map, CNN_map=CNN_map)

np.savez_compressed(f"{OUT}/dl_training_history.npz",
    mlp_loss=hist_mlp.history["loss"], mlp_val_loss=hist_mlp.history["val_loss"],
    cnn_loss=hist_cnn.history["loss"], cnn_val_loss=hist_cnn.history["val_loss"])

np.save(f"{OUT}/DL_MLP_pred_test.npy", pred_mlp_test)
np.save(f"{OUT}/DL_CNN_pred_test.npy", pred_cnn_test)

with open(f"{OUT}/dl_results.json", "w") as fjs:
    json.dump({
        "DL_MLP": dict(rmse=mlp_rmse, mae=mlp_mae, r2=mlp_r2, epochs=len(hist_mlp.history["loss"])),
        "DL_CNN": dict(rmse=cnn_rmse, mae=cnn_mae, r2=cnn_r2, epochs=len(hist_cnn.history["loss"])),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
    }, fjs, indent=2)

mlp.save(f"{OUT}/mlp_model.keras")
cnn.save(f"{OUT}/cnn_model.keras")
print("Saved DL_grid_predictions.npz, dl_training_history.npz, dl_results.json, models")
