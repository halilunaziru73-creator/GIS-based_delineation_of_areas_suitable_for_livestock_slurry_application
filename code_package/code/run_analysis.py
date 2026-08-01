import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from miniogr import read_shapefile
from geomtools import Grid, rasterize_features, euclid_buffer_mask, polygons_with_holes
from muniextract import extract_municipality
from proj_utm import lonlat_to_utm30n

DATA = '/home/claude/data'
OUT = '/home/claude/work/out'
os.makedirs(OUT, exist_ok=True)

t_start = time.time()
def log(msg):
    print(f'[{time.time()-t_start:7.1f}s] {msg}')

# ---------------------------------------------------------------
# 1. Reference grid from DEM_AOI.tif (already clipped to the 500 m AOI)
# ---------------------------------------------------------------
dem_img = Image.open(f'{DATA}/AOI/DEM_AOI.tif')
dem = np.array(dem_img).astype(np.float32)
n_rows, n_cols = dem.shape
PIXEL = 25.0
ORIGIN_X, ORIGIN_Y = 595700.0, 4669375.0   # from GeoTIFF tiepoint / .tfw
grid = Grid(ORIGIN_X, ORIGIN_Y, PIXEL, n_rows, n_cols)
dem_nodata = np.isnan(dem)
log(f'DEM grid: {n_rows} x {n_cols} px @ {PIXEL} m  | elev range '
    f'{np.nanmin(dem):.1f}-{np.nanmax(dem):.1f} m | bounds {grid.bounds()}')

# ---------------------------------------------------------------
# 2. Slope (%) and 3-class reclass  (Table 1, criterion 1)
# ---------------------------------------------------------------
dem_filled = np.where(dem_nodata, np.nan, dem)
dzdy, dzdx = np.gradient(dem_filled, PIXEL, PIXEL)
slope_pct = 100.0 * np.sqrt(dzdx**2 + dzdy**2)
slope_class = np.full((n_rows, n_cols), np.nan)
slope_class = np.where(slope_pct < 10, 2, np.where(slope_pct < 15, 1, 0)).astype(np.float32)
slope_class[dem_nodata] = np.nan
log(f'Slope computed. % suitable(<10%)={100*np.nanmean(slope_class[~dem_nodata]==2):.1f}% '
    f'limited={100*np.nanmean(slope_class[~dem_nodata]==1):.1f}% '
    f'excluded={100*np.nanmean(slope_class[~dem_nodata]==0):.1f}%')

# ---------------------------------------------------------------
# 3. Soil / lithology suitability  (Table 1, criterion 2) -- Suit_Class already in attrs
# ---------------------------------------------------------------
lith_feats = read_shapefile(f'{DATA}/AOI/Lithology_AOI.shp')
soil_class, soil_written = rasterize_features(lith_feats, grid, burn_field='Suit_Class',
                                               dtype=np.float32, combine='overwrite')
soil_class[~soil_written] = np.nan
log(f'Lithology rasterized: {len(lith_feats)} polygons, written {soil_written.mean()*100:.1f}% of grid')

# ---------------------------------------------------------------
# 4. Water protection (criterion 3): DPH riverbed + 5 m easement (already in DPH_AOI,
#    excluding the 100 m "Zona de policia") + wetlands/secondary streams buffered 5 m
# ---------------------------------------------------------------
dph_feats_all = read_shapefile(f'{DATA}/AOI/DPH_AOI.shp')
dph_feats = [f for f in dph_feats_all if f['attrs'].get('TIPO_ZONA') != 'Zona de Policía']
dph_raster, _ = rasterize_features(dph_feats, grid, default_value=1)
log(f'DPH (riverbed+servidumbre) features used: {len(dph_feats)}/{len(dph_feats_all)}')

franjas = read_shapefile(f"{DATA}/Surface water bodies/SGP2025_Franjas_Mun232.shp")
humedales = read_shapefile(f"{DATA}/Surface water bodies/SGP2025_Humedales_Mun232.shp")
wetstream_raster, _ = rasterize_features(franjas + humedales, grid, default_value=1)
wetstream_buf = euclid_buffer_mask(wetstream_raster > 0, PIXEL, 5.0)
water_protection_mask = (dph_raster > 0) | wetstream_buf
log(f'Water-protection exclusion (DPH+5m wetlands/streams buffer): '
    f'{100*water_protection_mask.mean():.1f}% of grid cells')

# ---------------------------------------------------------------
# 5. Urban buffer (criterion 4): 100 m around settlements (excl. "AAGR" agrarian class)
# ---------------------------------------------------------------
urban_feats_all = read_shapefile(f'{DATA}/AOI/UrbanArea_AOI.shp')
urban_feats = [f for f in urban_feats_all if f['attrs'].get('tipo') != 'AAGR']
urban_raster, _ = rasterize_features(urban_feats, grid, default_value=1)
urban_buffer_mask = euclid_buffer_mask(urban_raster > 0, PIXEL, 100.0)
log(f'Urban settlements used: {len(urban_feats)}/{len(urban_feats_all)}; '
    f'100 m buffer exclusion = {100*urban_buffer_mask.mean():.1f}% of grid cells')

# ---------------------------------------------------------------
# 6. Protected natural areas (criterion 5)
# ---------------------------------------------------------------
pna_feats = read_shapefile(f'{DATA}/AOI/PNA_AOI.shp')
pna_raster, _ = rasterize_features(pna_feats, grid, default_value=1)
pna_mask = pna_raster > 0
log(f'Protected natural area polygons: {len(pna_feats)}; exclusion = {100*pna_mask.mean():.1f}%')

# ---------------------------------------------------------------
# 7. Nitrate Vulnerable Zones (criterion 6)
# ---------------------------------------------------------------
nvz_feats = read_shapefile(f'{DATA}/AOI/NVZ_AOI.shp')
nvz_raster, _ = rasterize_features(nvz_feats, grid, default_value=1)
nvz_mask = nvz_raster > 0
log(f'NVZ polygons: {len(nvz_feats)}; exclusion = {100*nvz_mask.mean():.1f}%')

# ---------------------------------------------------------------
# 8. Land use / land cover eligibility (criterion 7): TA / PS via Suit_Class (2 = eligible)
# ---------------------------------------------------------------
lulc_feats = read_shapefile(f"{DATA}/Land use-Land cover/SGP2025_Mun232.shp")
lulc_class, lulc_written = rasterize_features(lulc_feats, grid, burn_field='Suit_Class',
                                               dtype=np.float32, combine='overwrite')
land_eligible = (lulc_class == 2)
log(f'LULC parcels: {len(lulc_feats)}; eligible TA/PS cover = {100*land_eligible.mean():.1f}% of grid, '
    f'parcel footprint covers {100*lulc_written.mean():.1f}% of grid')

tudela_feat = extract_municipality(
    f'{DATA}/Municipal boundary/recintos_municipales_inspire_peninbal_etrs89.shp',
    f'{DATA}/Municipal boundary/recintos_municipales_inspire_peninbal_etrs89.dbf',
    f'{DATA}/Municipal boundary/recintos_municipales_inspire_peninbal_etrs89.shx',
    'NAMEUNIT', 'Tudela')
tudela_feat['rings'] = [np.column_stack(lonlat_to_utm30n(r[:, 0], r[:, 1])) for r in tudela_feat['rings']]
tudela_raster, _ = rasterize_features([tudela_feat], grid, default_value=1)
tudela_mask = tudela_raster > 0
aoi_mask = ~dem_nodata
log(f'Tudela municipal boundary rasterized: area = {tudela_mask.sum()*PIXEL*PIXEL/1e6:.2f} km2 '
    f'(AOI incl. 500 m buffer = {aoi_mask.sum()*PIXEL*PIXEL/1e6:.2f} km2)')

non_suitable = water_protection_mask | urban_buffer_mask | pna_mask | nvz_mask | (~land_eligible)
non_suitable = non_suitable & aoi_mask
suitability_class = np.where(aoi_mask, np.where(non_suitable, 0, 1), np.nan).astype(np.float32)
log(f'Binary suitability (criteria 3-7 combined): suitable = '
    f'{100*np.nanmean(suitability_class[aoi_mask]):.1f}% of AOI')

soil_c = np.nan_to_num(soil_class, nan=0.0)
slope_c = np.nan_to_num(slope_class, nan=0.0)
suit_c = np.nan_to_num(suitability_class, nan=0.0)
result_aoi = suit_c * soil_c * slope_c
result_aoi = np.where(aoi_mask, result_aoi, np.nan)
result_tudela = np.where(tudela_mask, result_aoi, np.nan)

vals, counts = np.unique(result_aoi[tudela_mask & ~np.isnan(result_aoi)], return_counts=True)
log('Result classes within Tudela municipality (value: km2, %):')
tot = counts.sum()
for v, c in zip(vals, counts):
    log(f'   class {v:.0f}: {c*PIXEL*PIXEL/1e6:8.2f} km2  ({100*c/tot:5.1f}%)')

def area_km2(mask):
    return float((mask & tudela_mask).sum() * PIXEL * PIXEL / 1e6)

criteria_areas = {
    '1_slope_gt15pct'      : area_km2(slope_class == 0),
    '2_soil_unsuitable'    : area_km2(soil_class == 0),
    '3_water_protection_5m': area_km2(water_protection_mask),
    '4_urban_buffer_100m'  : area_km2(urban_buffer_mask),
    '5_protected_areas'    : area_km2(pna_mask),
    '6_nitrate_vulnerable' : area_km2(nvz_mask),
    '7_non_arable_landuse' : area_km2(~land_eligible),
}
log('Area individually affected by each constraint, within the municipality:')
for k, v in criteria_areas.items():
    log(f'   {k:24s}: {v:8.2f} km2  ({100*v/area_km2(tudela_mask):5.1f}%)')

log('Computing zonal statistics for arable/pasture parcels (this can take a while)...')
from skimage.draw import polygon as sk_polygon
zonal = []
for feat in lulc_feats:
    if feat['attrs'].get('Suit_Class') != 2:
        continue
    rings = feat['rings']
    if not rings:
        continue
    for exterior, holes in polygons_with_holes(rings):
        r_ext, c_ext = grid.world_to_pixel(exterior)
        r0 = max(0, int(np.floor(r_ext.min())))
        r1 = min(n_rows, int(np.ceil(r_ext.max())) + 1)
        c0 = max(0, int(np.floor(c_ext.min())))
        c1 = min(n_cols, int(np.ceil(c_ext.max())) + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        h, w = r1 - r0, c1 - c0
        rr, cc = sk_polygon(r_ext - r0, c_ext - c0, shape=(h, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[rr, cc] = True
        if mask.sum() == 0:
            continue
        sub = result_tudela[r0:r1, c0:c1]
        vals_ = sub[mask]
        vals_ = vals_[~np.isnan(vals_)]
        if len(vals_) == 0:
            continue
        zonal.append({
            'REFSIGPAC': feat['attrs'].get('REFSIGPAC'),
            'IDUSO25': feat['attrs'].get('IDUSO25'),
            'SUPERFICIE_m2': feat['attrs'].get('SUPERFICIE'),
            'mean_result': float(np.mean(vals_)),
            'n_cells': int(len(vals_)),
        })
log(f'Zonal statistics computed for {len(zonal)} TA/PS parcels')

import pandas as pd
zdf = pd.DataFrame(zonal)
zdf.to_csv(f'{OUT}/zonal_stats_TA_PS_parcels.csv', index=False)

np.savez_compressed(f'{OUT}/layers.npz',
    dem=dem, slope_pct=slope_pct, slope_class=slope_class, soil_class=soil_class,
    water_protection_mask=water_protection_mask, urban_buffer_mask=urban_buffer_mask,
    pna_mask=pna_mask, nvz_mask=nvz_mask, land_eligible=land_eligible,
    lulc_written=lulc_written, dem_nodata=dem_nodata, tudela_mask=tudela_mask,
    aoi_mask=aoi_mask, suitability_class=suitability_class, result_aoi=result_aoi,
    result_tudela=result_tudela)

with open(f'{OUT}/stats.json', 'w') as f:
    json.dump({
        'grid': {'n_rows': n_rows, 'n_cols': n_cols, 'pixel_m': PIXEL,
                 'origin_x': ORIGIN_X, 'origin_y': ORIGIN_Y},
        'area_tudela_km2': area_km2(tudela_mask),
        'area_aoi_km2': float(aoi_mask.sum()*PIXEL*PIXEL/1e6),
        'result_classes_km2': {str(int(v)): float(c*PIXEL*PIXEL/1e6) for v, c in zip(vals, counts)},
        'criteria_exclusion_km2': criteria_areas,
        'n_lulc_parcels': len(lulc_feats),
        'n_ta_ps_parcels_zonal': len(zonal),
    }, f, indent=2)
log('Done. Intermediate layers + stats saved to /home/claude/work/out')
