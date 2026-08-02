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
cov = np.load(f"{OUT}/covariates.npz")
aoi_mask = cov["aoi_mask"]

# ============================================================ FIGURE 5 =====
vg = np.load(f"{OUT}/variogram_residuals.npz")
lags, gamma, params = vg["lags"], vg["gamma"], vg["params"]
partial_sill, vrange, nugget = params

def spherical(h, psill, r, nug):
    h = np.asarray(h, dtype=float)
    out = np.where(h <= r, nug + psill*(1.5*(h/r) - 0.5*(h/r)**3), nug + psill)
    return out

hh = np.linspace(0, max(lags.max(), vrange*1.2), 200)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), dpi=300)
ax = axs[0]
ax.scatter(lags, gamma, color="#1f78b4", s=35, label="Empirical semivariance", zorder=3)
ax.plot(hh, spherical(hh, partial_sill, vrange, nugget), color="#e31a1c", lw=2,
        label="Fitted spherical model")
ax.axvline(vrange, color="gray", ls="--", lw=1)
ax.axhline(nugget + partial_sill, color="gray", ls=":", lw=1)
ax.text(vrange*1.02, 0.05*ax.get_ylim()[1], f"range \u2248 {vrange:.0f} m", fontsize=8)
ax.set_xlabel("Lag distance h (m)"); ax.set_ylabel("Semivariance \u03b3(h)")
ax.set_title("(a) Empirical and fitted variogram of OLS-trend residuals", fontsize=10, fontweight="bold", loc="left")
ax.legend(fontsize=8)
ax.annotate(f"nugget = {nugget:.1f}\npartial sill = {partial_sill:.1f}\nrange = {vrange:.0f} m\nnugget/sill = {nugget/(nugget+partial_sill):.2f}",
            xy=(0.55, 0.05), xycoords="axes fraction", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="gray"))

# residual histogram / spatial cloud
resid = vg["resid"]; xy = vg["xy"]
ax2 = axs[1]
sc = ax2.scatter(xy[:,0], xy[:,1], c=resid, cmap="RdBu_r", vmin=-np.abs(resid).max(), vmax=np.abs(resid).max(), s=22, edgecolor="k", linewidth=0.2)
cb = fig.colorbar(sc, ax=ax2, fraction=0.04)
cb.set_label("OLS residual (SI units)")
ax2.set_title("(b) Spatial distribution of trend residuals (training points)", fontsize=10, fontweight="bold", loc="left")
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_aspect("equal")
fig.suptitle("Figure 5. Variogram modelling of regression-kriging residuals", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig5_variogram.png", bbox_inches="tight")
plt.close(fig)
print("fig5 done")

# ============================================================ FIGURE 6 & 7 =
rk = np.load(f"{OUT}/RK_grid_prediction.npz")
SI_map, VAR_map = rk["SI_map"], rk["VAR_map"]

fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=300)
im0 = axs[0].imshow(SI_map, cmap="RdYlGn", extent=extent, origin="upper", vmin=0, vmax=100)
cb0 = fig.colorbar(im0, ax=axs[0], fraction=0.035, pad=0.02); cb0.set_label("Predicted SI (0\u2013100)")
style_map_axes(axs[0], "(a) Regression-kriging (RF trend + spherical-kriged residuals) prediction")
add_scalebar(axs[0], bounds, 3); add_north_arrow(axs[0], bounds)

im1 = axs[1].imshow(VAR_map, cmap="magma", extent=extent, origin="upper")
cb1 = fig.colorbar(im1, ax=axs[1], fraction=0.035, pad=0.02); cb1.set_label("Kriging variance (SI units$^2$)")
style_map_axes(axs[1], "(b) Ordinary-kriging prediction variance (uncertainty)")
add_scalebar(axs[1], bounds, 3); add_north_arrow(axs[1], bounds)

fig.suptitle("Figure 6. Regression-kriged suitability surface and associated prediction uncertainty", fontsize=11, y=1.0)
credit_text(fig, "RK-RF: random-forest trend on {slope, dist-water, dist-urban, dist-protected, NVZ, elevation} + ordinary kriging of residuals (spherical model).")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig6_RK_prediction_variance.png", bbox_inches="tight")
plt.close(fig)
print("fig6/7 done")
