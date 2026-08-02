import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/claude/work/pipeline")
from cartohelpers import add_scalebar, add_north_arrow, style_map_axes, credit_text

OUT = "/home/claude/work/out"
FIGS = f"{OUT}/figs"
meta = json.load(open(f"{OUT}/grid_meta.json"))
bounds = meta["bounds"]
extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

rk = np.load(f"{OUT}/RK_grid_prediction.npz")
dl = np.load(f"{OUT}/DL_grid_predictions.npz")
ref = np.load(f"{OUT}/mcda_reference.npz")
candidate_mask = ref["candidate_mask"]

RK_map = rk["SI_map"]; MLP_map = dl["MLP_map"]; CNN_map = dl["CNN_map"]

stack = np.stack([RK_map, MLP_map, CNN_map])
ensemble_mean = np.nanmean(stack, axis=0)
ensemble_std = np.nanstd(stack, axis=0)

# ============================================================ FIGURE 10 ====
fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.2), dpi=300)
maps = [("(a) RK-RF (regression kriging)", RK_map), ("(b) DL-MLP", MLP_map),
        ("(c) DL-CNN (patch-based)", CNN_map), ("(d) Three-model ensemble mean", ensemble_mean)]
for ax, (title, m) in zip(axs.ravel(), maps):
    im = ax.imshow(m, cmap="RdYlGn", extent=extent, origin="upper", vmin=0, vmax=100)
    style_map_axes(ax, title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("SI", fontsize=7)
fig.suptitle("Figure 10. Predicted suitability surfaces: regression kriging vs. deep-learning models", fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{FIGS}/fig10_model_comparison_maps.png", bbox_inches="tight")
plt.close(fig)
print("fig10 done")

# ============================================================ FIGURE 11 ====
fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), dpi=300)
im0 = axs[0].imshow(ensemble_mean, cmap="RdYlGn", extent=extent, origin="upper", vmin=0, vmax=100)
fig.colorbar(im0, ax=axs[0], fraction=0.035, pad=0.02).set_label("Ensemble mean SI")
style_map_axes(axs[0], "(a) Multi-model ensemble mean suitability")
add_scalebar(axs[0], bounds, 3); add_north_arrow(axs[0], bounds)

im1 = axs[1].imshow(ensemble_std, cmap="magma", extent=extent, origin="upper")
fig.colorbar(im1, ax=axs[1], fraction=0.035, pad=0.02).set_label("Inter-model SD (SI units)")
style_map_axes(axs[1], "(b) Inter-model disagreement (RK-RF vs. DL-MLP vs. DL-CNN)")
add_scalebar(axs[1], bounds, 3); add_north_arrow(axs[1], bounds)

fig.suptitle("Figure 11. Ensemble suitability map and structural (inter-model) uncertainty", fontsize=11.5, y=1.0)
credit_text(fig, "Ensemble = mean of RK-RF, DL-MLP and DL-CNN grid predictions; SD highlights zones of methodological disagreement.")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig11_ensemble_uncertainty.png", bbox_inches="tight")
plt.close(fig)
print("fig11 done")

# ============================================================ FIGURE 12 (feature importance) ====
imp = json.load(open(f"{OUT}/feature_importance.json"))
labels_map = {"slope_pct": "Slope (%)", "dist_water_m": "Distance to water (m)",
              "dist_urban_m": "Distance to urban areas (m)", "dist_protected_m": "Distance to protected areas (m)",
              "nvz": "Nitrate Vulnerable Zone (binary)", "elevation": "Elevation (m)"}
items = sorted(imp.items(), key=lambda kv: kv[1]["mean"])
names = [labels_map[k] for k, v in items]
means = [v["mean"] for k, v in items]
stds  = [v["std"] for k, v in items]

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
ypos = np.arange(len(names))
ax.barh(ypos, means, xerr=stds, color="#3182bd", edgecolor="black", capsize=3)
ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("Permutation importance (\u0394R\u00b2, mean \u00b1 SD over 30 repeats)")
ax.set_title("Figure 12. Covariate importance for the random-forest trend model", fontsize=10.5, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig12_feature_importance.png", bbox_inches="tight")
plt.close(fig)
print("fig12 done")
