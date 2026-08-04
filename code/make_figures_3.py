import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent.parent
OUT = str(BASE / "outputs_data")
FIGS = f"{OUT}/figs"

hist = np.load(f"{OUT}/dl_training_history.npz")
dl_res = json.load(open(f"{OUT}/dl_results.json"))

fig = plt.figure(figsize=(12.5, 8.6), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])

# ---- (a) MLP architecture schematic ----
ax1 = fig.add_subplot(gs[0, 0]); ax1.axis("off")
layer_sizes = [8, 64, 32, 16, 1]
layer_names = ["Inputs\n(8 covariates)", "Dense 64\nReLU + BN", "Dense 32\nReLU", "Dense 16\nReLU", "Output\n(SI)"]
xs = np.linspace(0.05, 0.95, len(layer_sizes))
maxn = 10
for li, (x, n) in enumerate(zip(xs, layer_sizes)):
    nn = min(n, maxn)
    ys = np.linspace(0.15, 0.85, nn)
    for y in ys:
        ax1.add_patch(mpatches.Circle((x, y), 0.018, color="#3182bd" if li not in (0, len(layer_sizes)-1) else "#31a354"))
    if li < len(layer_sizes)-1:
        nn2 = min(layer_sizes[li+1], maxn)
        ys2 = np.linspace(0.15, 0.85, nn2)
        for y in ys:
            for y2 in ys2:
                ax1.plot([x+0.018, xs[li+1]-0.018], [y, y2], color="grey", lw=0.25, alpha=0.5, zorder=0)
    ax1.text(x, 0.95, layer_names[li], ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.05)
ax1.set_title("(a) DL-MLP architecture: dense feed-forward network on point covariates", fontsize=9.5, fontweight="bold", loc="left")

# ---- (b) CNN architecture schematic ----
ax2 = fig.add_subplot(gs[0, 1]); ax2.axis("off")
blocks = [("Input\n9x9x6", "#8856a7"), ("Conv3x3\n16 filt.", "#3182bd"), ("Conv3x3\n32 filt.", "#3182bd"),
          ("MaxPool\n2x2", "#969696"), ("Conv3x3\n32 filt.", "#3182bd"), ("GlobalAvgPool", "#969696"),
          ("Dense 32\nReLU", "#31a354"), ("Output\n(SI)", "#e34a33")]
xs2 = np.linspace(0.06, 0.94, len(blocks))
w = 0.09
for x, (lbl, col) in zip(xs2, blocks):
    ax2.add_patch(mpatches.FancyBboxPatch((x-w/2, 0.42), w, 0.22, boxstyle="round,pad=0.01",
                                            fc=col, ec="black", lw=0.8, alpha=0.85))
    ax2.text(x, 0.53, lbl, ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
for i in range(len(xs2)-1):
    ax2.annotate("", xy=(xs2[i+1]-w/2, 0.53), xytext=(xs2[i]+w/2, 0.53),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1))
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.set_title("(b) DL-CNN architecture: 9\u00d79 multichannel patch \u2192 convolutional feature extractor", fontsize=9.5, fontweight="bold", loc="left")
ax2.text(0.5, 0.15, "Input channels: slope, dist-to-water, dist-to-urban, dist-to-protected, NVZ indicator, elevation",
         ha="center", fontsize=7.2, style="italic")

# ---- (c) training curves ----
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(hist["mlp_loss"], color="#1f78b4", label="Training loss (MSE)")
ax3.plot(hist["mlp_val_loss"], color="#e31a1c", label="Validation loss (MSE)")
ax3.set_xlabel("Epoch"); ax3.set_ylabel("MSE (SI units$^2$)")
ax3.set_title(f"(c) DL-MLP learning curves (early stop @ epoch {dl_res['DL_MLP']['epochs']})",
              fontsize=9.5, fontweight="bold", loc="left")
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(hist["cnn_loss"], color="#1f78b4", label="Training loss (MSE)")
ax4.plot(hist["cnn_val_loss"], color="#e31a1c", label="Validation loss (MSE)")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("MSE (SI units$^2$)")
ax4.set_title(f"(d) DL-CNN learning curves (early stop @ epoch {dl_res['DL_CNN']['epochs']})",
              fontsize=9.5, fontweight="bold", loc="left")
ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

fig.suptitle("Figure 7. Deep-learning model architectures and training dynamics", fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{FIGS}/fig7_DL_architectures_training.png", bbox_inches="tight")
plt.close(fig)
print("fig7 done")
