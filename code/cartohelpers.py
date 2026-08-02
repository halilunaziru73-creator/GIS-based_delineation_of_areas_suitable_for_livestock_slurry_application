import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.font_manager import FontProperties

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#333333",
    "figure.facecolor": "white",
})

def add_scalebar(ax, bounds, length_km=2, loc=(0.03, 0.04)):
    x0, x1 = bounds[0], bounds[2]
    y0, y1 = bounds[1], bounds[3]
    w = x1 - x0; h = y1 - y0
    sb_x = x0 + loc[0]*w
    sb_y = y0 + loc[1]*h
    L = length_km * 1000.0
    ax.plot([sb_x, sb_x+L], [sb_y, sb_y], color="black", lw=2.5, solid_capstyle="butt",
            transform=ax.transData, zorder=10)
    for frac, txt in [(0, "0"), (1, f"{length_km} km")]:
        ax.text(sb_x+frac*L, sb_y + 0.012*h, txt, ha="center", va="bottom", fontsize=7, zorder=10)

def add_north_arrow(ax, bounds, loc=(0.93, 0.90)):
    x0, x1 = bounds[0], bounds[2]; y0, y1 = bounds[1], bounds[3]
    w = x1-x0; h = y1-y0
    ax_x = x0 + loc[0]*w; ax_y = y0 + loc[1]*h
    arr_len = 0.06*h
    ax.annotate("N", xy=(ax_x, ax_y+arr_len), xytext=(ax_x, ax_y),
                arrowprops=dict(facecolor="black", width=3.5, headwidth=10, headlength=8),
                ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=10)

def style_map_axes(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")

def credit_text(fig, text):
    fig.text(0.01, 0.01, text, fontsize=6.5, color="#555555")
