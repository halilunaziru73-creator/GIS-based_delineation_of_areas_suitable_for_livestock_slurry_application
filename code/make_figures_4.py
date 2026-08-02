import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "/home/claude/work/out"
FIGS = f"{OUT}/figs"

df = pd.read_csv(f"{OUT}/sample_points.csv")
test = df[df.split == "test"].reset_index(drop=True)
yte = test["SI_obs"].values

preds = {
    "RK-LM": np.load(f"{OUT}/RK_LM_pred_test.npy"),
    "RK-RF": np.load(f"{OUT}/RK_RF_pred_test.npy"),
    "DL-MLP": np.load(f"{OUT}/DL_MLP_pred_test.npy"),
    "DL-CNN": np.load(f"{OUT}/DL_CNN_pred_test.npy"),
}
rk_res = json.load(open(f"{OUT}/rk_results.json"))["results"]
dl_res = json.load(open(f"{OUT}/dl_results.json"))
metrics = {"RK-LM": rk_res["RK_LM"], "RK-RF": rk_res["RK_RF"],
           "DL-MLP": dl_res["DL_MLP"], "DL-CNN": dl_res["DL_CNN"]}

# ============================================================ FIGURE 8 =====
fig, axs = plt.subplots(1, 4, figsize=(15, 3.7), dpi=300, sharex=True, sharey=True)
colors = {"RK-LM": "#7570b3", "RK-RF": "#1b9e77", "DL-MLP": "#d95f02", "DL-CNN": "#e7298a"}
for ax, (name, pred) in zip(axs, preds.items()):
    ax.scatter(yte, pred, s=22, alpha=0.75, color=colors[name], edgecolor="black", linewidth=0.2)
    lims = [0, 100]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    m = metrics[name]
    ax.set_title(f"{name}\nRMSE={m['rmse']:.2f}  R\u00b2={m['r2']:.2f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Observed SI (held-out)")
    ax.set_aspect("equal")
axs[0].set_ylabel("Predicted SI")
fig.suptitle("Figure 8. Observed vs. predicted suitability index on the spatially held-out test blocks (n=%d)" % len(yte),
             fontsize=11, y=1.06)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig8_obs_vs_pred_scatter.png", bbox_inches="tight")
plt.close(fig)
print("fig8 done")

# ============================================================ FIGURE 9 (Taylor diagram) =====
val = json.load(open(f"{OUT}/validation_extra.json"))["taylor_stats"]
ref_std = val["Reference"]["std"]

fig = plt.figure(figsize=(7.2, 6.6), dpi=300)
ax = fig.add_subplot(111, polar=True)
ax.set_thetamin(0); ax.set_thetamax(90)
ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
corr_ticks = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
ax.set_thetagrids([np.degrees(np.arccos(c)) for c in corr_ticks],
                   labels=[str(c) for c in corr_ticks], fontsize=7.5)
max_std = max(ref_std, *[v["std"] for k, v in val.items() if k != "Reference"]) * 1.25
ax.set_rlim(0, max_std)
ax.set_rlabel_position(135)
ax.set_ylabel("")
ax.text(np.radians(45), max_std*1.12, "Correlation coefficient", ha="center", fontsize=9, fontweight="bold")

# reference point
theta_ref = np.arccos(1.0)
ax.plot([theta_ref], [ref_std], marker="*", color="black", markersize=18, label="Reference (observed SI)")

# RMSE arcs (centred pattern RMS)
theta_grid = np.linspace(0, np.pi/2, 100)
for rms in np.linspace(max_std*0.25, max_std*1.0, 4):
    xs = ref_std + rms*np.cos(theta_grid)
    ys = rms*np.sin(theta_grid)
    rr = np.sqrt(xs**2 + ys**2); th = np.arctan2(ys, xs)
    ax.plot(th, rr, color="grey", ls=":", lw=0.6)

colors2 = {"RK-LM": "#7570b3", "RK-RF": "#1b9e77", "DL-MLP": "#d95f02", "DL-CNN": "#e7298a"}
name_map = {"RK-LM": "RK_LM", "RK-RF": "RK_RF", "DL-MLP": "DL_MLP", "DL-CNN": "DL_CNN"}
for name, col in colors2.items():
    v = val[name_map[name]]
    theta = np.arccos(np.clip(v["corr"], -1, 1))
    ax.plot([theta], [v["std"]], marker="o", markersize=11, color=col, label=name,
            markeredgecolor="black", markeredgewidth=0.6)

ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8.5, frameon=True)
ax.set_title("Figure 9. Taylor diagram \u2014 model performance on held-out test blocks", fontsize=11, pad=30)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig9_taylor_diagram.png", bbox_inches="tight")
plt.close(fig)
print("fig9 done")
