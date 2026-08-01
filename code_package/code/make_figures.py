import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

OUT = '/home/claude/work/out'
FIG = '/home/claude/work/out/figs'
os.makedirs(FIG, exist_ok=True)

AUTHOR = 'Naziru Halilu'

d = np.load(f'{OUT}/layers.npz')
stats = json.load(open(f'{OUT}/stats.json'))
PIXEL = stats['grid']['pixel_m']
ox, oy = stats['grid']['origin_x'], stats['grid']['origin_y']
n_rows, n_cols = stats['grid']['n_rows'], stats['grid']['n_cols']
extent = [ox, ox + n_cols*PIXEL, oy - n_rows*PIXEL, oy]  # left,right,bottom,top
data_w = extent[1] - extent[0]
data_h = extent[3] - extent[2]

aoi_mask = d['aoi_mask']
tudela_mask = d['tudela_mask']

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})

def tudela_outline():
    from skimage import measure
    return measure.find_contours(tudela_mask.astype(float), 0.5)

# ------------------------------------------------------------------
# Layout geometry: the map panel reserves a small top/side margin and
# a slightly larger bottom margin (for scale bar + source + author),
# exactly like the "map / legend / scale bar / north arrow / metadata"
# item layout of a QGIS Print Layout composition.
# ------------------------------------------------------------------
PAD_TOP    = 0.07
PAD_SIDE   = 0.035
PAD_BOTTOM = 0.20

def _set_map_extent(ax):
    ax.set_xlim(extent[0] - PAD_SIDE*data_w, extent[1] + PAD_SIDE*data_w)
    ax.set_ylim(extent[2] - PAD_BOTTOM*data_h, extent[3] + PAD_TOP*data_h)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.1)

def _draw_boundary(ax):
    for cont in tudela_outline():
        xs = ox + cont[:, 1]*PIXEL
        ys = oy - cont[:, 0]*PIXEL
        ax.plot(xs, ys, color='black', linewidth=1.0, zorder=5)

def north_arrow(ax):
    x = extent[1] - 0.10*data_w
    y0 = extent[3] + 0.015*data_h
    y1 = extent[3] + PAD_TOP*data_h - 0.008*data_h
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(facecolor='black', edgecolor='black', width=3.6, headwidth=12, headlength=10),
                zorder=6)
    ax.text(x, y1 + 0.006*data_h, 'N', ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=6)

def scale_bar(ax, length_km=5):
    bar_y = extent[2] - 0.065*data_h
    x0 = extent[0]
    bh = 0.013*data_h
    for i in range(2):
        ax.add_patch(Rectangle((x0 + i*length_km*1000, bar_y), length_km*1000, bh,
                                facecolor='black' if i == 0 else 'white',
                                edgecolor='black', linewidth=0.7, zorder=6))
    for i, lab in enumerate(['0', f'{length_km}', f'{2*length_km} km']):
        ax.text(x0 + i*length_km*1000, bar_y - 0.016*data_h, lab, fontsize=6.5, ha='center', va='top', zorder=6)

def source_author(ax, source_text=None):
    y_author = extent[2] - 0.140*data_h
    ax.text(extent[1], y_author, f'Author: {AUTHOR}', fontsize=7.5, ha='right', va='top',
            style='italic', color='#333333', zorder=6)

def legend_panel(ax_leg, colors, labels, title=None):
    """Dedicated, bordered legend panel to the side of the map (like a
    QGIS layout Legend item), instead of floating over mapped data."""
    ax_leg.axis('off')
    handles = []
    for c in colors:
        if c == 'line':
            handles.append(Line2D([0], [0], color='black', linewidth=1.2))
        else:
            handles.append(Patch(facecolor=c, edgecolor='black', linewidth=0.5))
    leg = ax_leg.legend(handles, labels, title=title, loc='center left',
                         fontsize=8, title_fontsize=9, frameon=True,
                         framealpha=1.0, edgecolor='black', borderpad=1.0,
                         handlelength=1.6, handleheight=1.3, labelspacing=1.0,
                         bbox_to_anchor=(0.0, 0.5))
    leg.get_frame().set_linewidth(1.0)

def map_with_legend(figsize, title, legend_colors=None, legend_labels=None, legend_title=None):
    """Two-panel QGIS-style layout: map (left) + legend box (right)."""
    fig = plt.figure(figsize=figsize)
    if legend_colors is not None:
        gs = gridspec.GridSpec(1, 2, width_ratios=[3.4, 1.05], wspace=0.02,
                                left=0.045, right=0.97, top=0.90, bottom=0.03, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])
        legend_panel(ax_leg, legend_colors, legend_labels, legend_title)
    else:
        gs = gridspec.GridSpec(1, 1, left=0.06, right=0.96, top=0.90, bottom=0.03, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
    _set_map_extent(ax)
    _draw_boundary(ax)
    fig.suptitle(title, fontsize=11.5, y=0.965)
    return fig, ax

def finish(fig, ax, path, source_text, scale_km=5):
    north_arrow(ax)
    scale_bar(ax, scale_km)
    source_author(ax, source_text)
    fig.savefig(path, dpi=220, facecolor='white')
    plt.close(fig)

SRC_DEM   = 'CNIG LiDAR MDT25 (25 m)'
SRC_SOIL  = 'IDENA 1:25 000 geological map'
SRC_WATER = 'MITECO DPH; SIGPAC wetlands/streams'
SRC_URBAN = 'CNIG Nomenclator settlements'
SRC_PNA   = 'EEA Natura 2000 / CDDA'
SRC_NVZ   = 'MITECO Nitrate Vulnerable Zones'
SRC_LULC  = 'SIGPAC land use / land cover'
SRC_ALL   = 'DEM: CNIG; lithology: IDENA; hydrology/NVZ: MITECO; urban: CNIG; protected areas: EEA; land use: SIGPAC'

CMAP3 = ['#d01c8b', '#f1b6da', '#4dac26']

# ------------------------------------------------------------------
# Fig 1: DEM, classed elevation with legend panel
# ------------------------------------------------------------------
elev_bins = [240, 300, 350, 400, 500, 780]
elev_colors = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#d73027']
elev_labels = ['240-300', '300-350', '350-400', '400-500', '500-767']
dem_arr = d['dem']
elev_class = np.full(dem_arr.shape, np.nan)
for i in range(len(elev_bins)-1):
    m = (dem_arr >= elev_bins[i]) & (dem_arr < elev_bins[i+1] + (1 if i == len(elev_bins)-2 else 0))
    elev_class[m] = i
elev_class_m = np.ma.masked_invalid(elev_class)
cmapE = ListedColormap(elev_colors)
normE = BoundaryNorm(np.arange(-0.5, len(elev_colors), 1), cmapE.N)

fig, ax = map_with_legend((9.2, 7.6), 'Fig. 1. Digital Elevation Model (25 m), Tudela municipality',
                           elev_colors, elev_labels, 'Elevation (m a.s.l.)')
ax.imshow(elev_class_m, extent=extent, cmap=cmapE, norm=normE, zorder=2)
finish(fig, ax, f'{FIG}/fig1_dem.png', SRC_DEM)

# ------------------------------------------------------------------
# Fig 2: Slope classes
# ------------------------------------------------------------------
fig, ax = map_with_legend((9.2, 7.6), 'Fig. 2. Slope suitability classes (from DEM)',
                           CMAP3[::-1], ['Suitable (<10%)', 'Limited (10-15%)', 'Excluded (>=15%)'], 'Slope class')
slope_class = np.ma.masked_invalid(d['slope_class'])
cmap = ListedColormap(CMAP3)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
ax.imshow(slope_class, extent=extent, cmap=cmap, norm=norm, zorder=2)
finish(fig, ax, f'{FIG}/fig2_slope_class.png', SRC_DEM)

# ------------------------------------------------------------------
# Fig 3: Soil / lithology suitability
# ------------------------------------------------------------------
fig, ax = map_with_legend((9.2, 7.6), 'Fig. 3. Soil / lithology suitability classes',
                           CMAP3[::-1], ['Suitable', 'Suitable w/ conditions', 'Non-suitable'], 'Soil class')
soil_class = np.ma.masked_invalid(d['soil_class'])
ax.imshow(soil_class, extent=extent, cmap=cmap, norm=norm, zorder=2)
finish(fig, ax, f'{FIG}/fig3_soil_class.png', SRC_SOIL)

# ------------------------------------------------------------------
# Fig 4: individual exclusion layers - 2x2 map panels + ONE shared legend column
# ------------------------------------------------------------------
excl_color = '#b8103f'
fig = plt.figure(figsize=(11.5, 9.6))
gs = gridspec.GridSpec(2, 3, width_ratios=[3, 3, 1.15], wspace=0.05, hspace=0.16,
                        left=0.035, right=0.975, top=0.90, bottom=0.045, figure=fig)
panels = [
    ('water_protection_mask', 'a) Water protection buffer\n(DPH + 5 m wetlands/streams)', 0, 0),
    ('urban_buffer_mask', 'b) Urban exclusion buffer (100 m)', 0, 1),
    ('pna_mask', 'c) Protected natural areas\n(Natura 2000 / LIC / ZEPA)', 1, 0),
    ('nvz_mask', 'd) Nitrate Vulnerable Zones (NVZ)', 1, 1),
]
for key, ptitle, r, cidx in panels:
    ax = fig.add_subplot(gs[r, cidx])
    _set_map_extent(ax)
    arr = d[key].astype(float)
    arr = np.where(aoi_mask, arr, np.nan)
    arr = np.ma.masked_invalid(arr)
    ax.imshow(arr, extent=extent, cmap=ListedColormap(['#f2f2f2', excl_color]), vmin=0, vmax=1, zorder=2)
    _draw_boundary(ax)
    ax.set_title(ptitle, fontsize=9)
    north_arrow(ax)
    scale_bar(ax, 5)
ax_leg = fig.add_subplot(gs[:, 2])
legend_panel(ax_leg, ['#f2f2f2', excl_color], ['Not affected', 'Excluded'], 'Exclusion status')
fig.suptitle('Fig. 4. Individual regulatory/environmental exclusion layers', y=0.965, fontsize=13)
fig.text(0.975, 0.012, f'Author: {AUTHOR}', fontsize=8, ha='right', style='italic', color='#333333')
fig.savefig(f'{FIG}/fig4_exclusion_layers.png', dpi=220, facecolor='white')
plt.close(fig)

# ------------------------------------------------------------------
# Fig 5: Land use eligibility
# ------------------------------------------------------------------
fig, ax = map_with_legend((9.2, 7.6), 'Fig. 5. SIGPAC land-use eligibility for slurry spreading',
                           ['#b8e186', '#8e5a2a'], ['Arable land (TA) / Pasture (PS)', 'Other land cover'], 'Land use')
land = d['land_eligible'].astype(float)
land = np.where(d['lulc_written'], land, np.nan)
land = np.ma.masked_invalid(land)
ax.imshow(land, extent=extent, cmap=ListedColormap(['#8e5a2a', '#b8e186']), vmin=0, vmax=1, zorder=2)
finish(fig, ax, f'{FIG}/fig5_landuse.png', SRC_LULC)

# ------------------------------------------------------------------
# Fig 6: Binary suitability (criteria 3-7 merged)
# ------------------------------------------------------------------
fig, ax = map_with_legend((9.2, 7.6), 'Fig. 6. Binary suitability mask (regulatory criteria 3-7 combined)',
                           ['#4dac26', '#d01c8b'], ['Suitable', 'Non-suitable'], 'Suitability')
sc = np.ma.masked_invalid(d['suitability_class'])
ax.imshow(sc, extent=extent, cmap=ListedColormap(['#d01c8b', '#4dac26']), vmin=0, vmax=1, zorder=2)
finish(fig, ax, f'{FIG}/fig6_suitability_binary.png', SRC_ALL)

# ------------------------------------------------------------------
# Fig 7: Final 4-class result map (key output)
# ------------------------------------------------------------------
fig, ax = map_with_legend((9.6, 8.0), 'Fig. 7. Final slurry-application suitability map, Tudela municipality',
                           ['#4dac26', '#b8e186', '#f1b6da', '#d01c8b'],
                           ['Optimal location', 'Partial constraints', 'High constraints', 'Non suitable'],
                           'Suitability class')
result = np.ma.masked_invalid(d['result_tudela'])
cmap4 = ListedColormap(['#d01c8b', '#f1b6da', '#b8e186', '#4dac26'])
norm4 = BoundaryNorm([-0.5, 0.5, 1.5, 3, 4.5], cmap4.N)
ax.imshow(result, extent=extent, cmap=cmap4, norm=norm4, zorder=2)
finish(fig, ax, f'{FIG}/fig7_result_tudela.png', SRC_ALL)

# ------------------------------------------------------------------
# Fig 8: bar chart, area by final class  (statistical chart - not a map;
# kept clean and publication-styled, with source/author credit line)
# ------------------------------------------------------------------
classes_km2 = stats['result_classes_km2']
labels_map = {'0': 'Non suitable', '1': 'High\nconstraints', '2': 'Partial\nconstraints', '4': 'Optimal\nlocation'}
order = ['0', '1', '2', '4']
vals = [classes_km2.get(k, 0) for k in order]
colors_bar = ['#d01c8b', '#f1b6da', '#b8e186', '#4dac26']
fig, ax = plt.subplots(figsize=(6.8, 5.2))
bars = ax.bar([labels_map[k] for k in order], vals, color=colors_bar, edgecolor='black')
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+2, f'{v:.1f} km2\n({100*v/sum(vals):.1f}%)',
            ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Area (km2)')
ax.set_title('Fig. 8. Land area by slurry-application suitability class\n(Tudela municipality)')
ax.set_ylim(0, max(vals)*1.28)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.text(0.98, 0.015, f'Author: {AUTHOR}', fontsize=8, ha='right', style='italic', color='#333333')
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(f'{FIG}/fig8_bar_classes.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------
# Fig 9: horizontal bars, area excluded by each individual criterion
# ------------------------------------------------------------------
crit = stats['criteria_exclusion_km2']
names = {
    '1_slope_gt15pct': 'Slope >= 15%',
    '2_soil_unsuitable': 'Non-suitable soils/lithology',
    '3_water_protection_5m': 'Water protection (DPH + 5 m buffer)',
    '4_urban_buffer_100m': 'Urban buffer (100 m)',
    '5_protected_areas': 'Protected natural areas',
    '6_nitrate_vulnerable': 'Nitrate Vulnerable Zone',
    '7_non_arable_landuse': 'Non-arable/pasture land cover',
}
keys = list(crit.keys())
vals = [crit[k] for k in keys]
labels = [names[k] for k in keys]
order_idx = np.argsort(vals)
fig, ax = plt.subplots(figsize=(7.8, 5.2))
ax.barh([labels[i] for i in order_idx], [vals[i] for i in order_idx],
        color='#7570b3', edgecolor='black')
for i, idx in enumerate(order_idx):
    ax.text(vals[idx]+1, i, f'{vals[idx]:.1f} km2 ({100*vals[idx]/stats["area_tudela_km2"]:.0f}%)',
            va='center', fontsize=8)
ax.set_xlabel('Area individually affected (km2, may overlap)')
ax.set_title('Fig. 9. Municipal area affected by each individual exclusion criterion')
ax.set_xlim(0, max(vals)*1.22)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.text(0.98, 0.015, f'Author: {AUTHOR}', fontsize=8, ha='right', style='italic', color='#333333')
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(f'{FIG}/fig9_criteria_bars.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------
# Fig 10: histogram of parcel-level mean suitability
# ------------------------------------------------------------------
import pandas as pd
zdf = pd.read_csv(f'{OUT}/zonal_stats_TA_PS_parcels.csv')
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.hist(zdf['mean_result'], bins=30, color='#4dac26', edgecolor='black', alpha=0.85)
ax.set_xlabel('Parcel mean suitability score (0-4)')
ax.set_ylabel('Number of arable/pasture parcels')
ax.set_title(f'Fig. 10. Distribution of parcel-level mean suitability\n(n = {len(zdf)} SIGPAC TA/PS parcels)')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.text(0.98, 0.015, f'Author: {AUTHOR}', fontsize=8, ha='right', style='italic', color='#333333')
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(f'{FIG}/fig10_parcel_hist.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------
# Fig 11: context map (hillshade + hydrology + urban), with legend panel
# ------------------------------------------------------------------
from scipy.ndimage import sobel
dem_f = np.nan_to_num(d['dem'], nan=np.nanmean(d['dem']))
dx = sobel(dem_f, axis=1)/8; dy = sobel(dem_f, axis=0)/8
slope_rad = np.arctan(np.hypot(dx, dy)/PIXEL)
aspect = np.arctan2(-dx, dy)
az, alt = np.radians(315), np.radians(45)
hillshade = (np.sin(alt)*np.cos(slope_rad) + np.cos(alt)*np.sin(slope_rad)*np.cos(az-aspect))
hillshade = np.ma.masked_where(~aoi_mask, hillshade)

fig, ax = map_with_legend((9.8, 8.0),
    'Fig. 11. Study-area context: hillshaded relief, hydrology\nand urban settlements, municipality of Tudela (Navarre, Spain)',
    ['#2c7fb8', '#636363', 'line'],
    ['Water protection buffer', 'Urban settlements + 100 m buffer', 'Tudela municipal boundary'],
    None)
ax.imshow(hillshade, extent=extent, cmap='gray', vmin=-1, vmax=1, alpha=0.6, zorder=1)
ax.imshow(np.ma.masked_where(~d['water_protection_mask'], np.ones_like(d['dem'])),
          extent=extent, cmap=ListedColormap(['#2c7fb8']), vmin=0, vmax=1, zorder=2)
ax.imshow(np.ma.masked_where(~(d['urban_buffer_mask'] & ~d['water_protection_mask']), np.ones_like(d['dem'])*0.5),
          extent=extent, cmap=ListedColormap(['#636363']), vmin=0, vmax=1, alpha=0.6, zorder=2)
finish(fig, ax, f'{FIG}/fig11_context.png', f'{SRC_DEM}; {SRC_WATER}; {SRC_URBAN}')

print('All figures re-written as QGIS-style publication layouts to', FIG)
for f in sorted(os.listdir(FIG)):
    print(' -', f)
