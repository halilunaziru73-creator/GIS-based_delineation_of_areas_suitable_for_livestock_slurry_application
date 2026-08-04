"""
Stage 1: Build a 25 m raster covariate stack for the Tudela AOI from the
supplied vector/raster layers (all already clipped, EPSG:25830).
"""
import os, json
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
DATA = str(BASE / "source_data")
OUT  = str(BASE / "outputs_data")
os.makedirs(OUT, exist_ok=True)

PIXEL = 25.0

# ---- reference grid from DEM_AOI.tif -------------------------------------
with rasterio.open(f"{DATA}/AOI/AOI/DEM_AOI.tif") as src:
    dem = src.read(1).astype(np.float32)
    transform = src.transform
    crs = src.crs
    height, width = src.height, src.width
    bounds = src.bounds

dem_nodata = ~np.isfinite(dem)
print(f"Grid: {width} x {height} px @ {PIXEL} m | elev range "
      f"{np.nanmin(dem):.1f}-{np.nanmax(dem):.1f} m")

def rasterize_gdf(gdf, transform=transform, out_shape=(height, width), fill=0, value=1):
    if len(gdf) == 0:
        return np.zeros(out_shape, dtype=np.uint8)
    shapes = [(geom, value) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    return rasterize(shapes, out_shape=out_shape, transform=transform, fill=fill, dtype=np.uint8)

def dist_to_mask(mask):
    # distance in metres from every cell to nearest True cell in mask
    if mask.sum() == 0:
        return np.full(mask.shape, np.nan, dtype=np.float32)
    inv = ~mask
    d = distance_transform_edt(inv, sampling=(PIXEL, PIXEL))
    return d.astype(np.float32)

# ---- 1. Slope (%) from DEM ------------------------------------------------
dem_filled = np.where(dem_nodata, np.nan, dem)
dzdy, dzdx = np.gradient(dem_filled, PIXEL, PIXEL)
slope_pct = (100.0 * np.sqrt(dzdx**2 + dzdy**2)).astype(np.float32)

# ---- 2. Soil / lithology suitability (rasterize Suit_Class, 0/2) ---------
lith = gpd.read_file(f"{DATA}/AOI/AOI/Lithology_AOI.shp")
soil_suit01 = np.zeros((height, width), dtype=np.uint8)
for cls_val, sub in lith.groupby("Suit_Class"):
    if cls_val is None:
        continue
    r = rasterize_gdf(sub, value=int(cls_val))
    soil_suit01 = np.where(r > 0, r, soil_suit01)
soil_class01 = (soil_suit01 == 2).astype(np.uint8)  # 1 = suitable soil

# ---- 3. Water protection layers (DPH minus 'Zona de Policia', + 5 m wetlands/streams buffer)
dph = gpd.read_file(f"{DATA}/AOI/AOI/DPH_AOI.shp")
dph_use = dph[dph["TIPO_ZONA"] != "Zona de Policía"]
dph_mask = rasterize_gdf(dph_use) > 0

franjas = gpd.read_file(f"{DATA}/data/Surface_water_bodies/Surface water bodies/SGP2025_Franjas_Mun232.shp") if os.path.exists(f"{DATA}/data") else None
franjas = gpd.read_file(f"{DATA}/Surface_water_bodies/Surface water bodies/SGP2025_Franjas_Mun232.shp")
humedales = gpd.read_file(f"{DATA}/Surface_water_bodies/Surface water bodies/SGP2025_Humedales_Mun232.shp")
wetstream_raw = rasterize_gdf(franjas) | rasterize_gdf(humedales)
wetstream_buf = dist_to_mask(wetstream_raw > 0) <= 5.0
water_mask = dph_mask | wetstream_buf
dist_water = dist_to_mask(dph_mask | (wetstream_raw > 0))

# ---- 4. Urban settlement buffer (100 m, excl. 'AAGR') ---------------------
urban = gpd.read_file(f"{DATA}/AOI/AOI/UrbanArea_AOI.shp")
urban_use = urban[urban["tipo"] != "AAGR"] if "tipo" in urban.columns else urban
urban_raw = rasterize_gdf(urban_use) > 0
dist_urban = dist_to_mask(urban_raw)
urban_buffer_mask = dist_urban <= 100.0

# ---- 5. Protected Natural Areas -------------------------------------------
pna = gpd.read_file(f"{DATA}/AOI/AOI/PNA_AOI.shp")
pna_mask = rasterize_gdf(pna) > 0
dist_protected = dist_to_mask(pna_mask)

# ---- 6. Nitrate Vulnerable Zones ------------------------------------------
nvz = gpd.read_file(f"{DATA}/AOI/AOI/NVZ_AOI.shp")
nvz_mask = rasterize_gdf(nvz) > 0
dist_nvz_boundary = dist_to_mask(nvz_mask) if nvz_mask.sum() < nvz_mask.size else np.zeros_like(dem)

# ---- 7. Land use / land cover eligibility (SIGPAC Suit_Class == 2) --------
lulc = gpd.read_file(f"{DATA}/Land_use_Land_cover/Land use-Land cover/SGP2025_Mun232.shp")
land_written = np.zeros((height, width), dtype=np.uint8)
land_elig = np.zeros((height, width), dtype=np.uint8)
for cls_val, sub in lulc.groupby("Suit_Class"):
    if cls_val is None:
        continue
    r = rasterize_gdf(sub, value=1)
    land_written = np.where(r > 0, 1, land_written)
    if int(cls_val) == 2:
        land_elig = np.where(r > 0, 1, land_elig)
land_eligible = land_elig.astype(bool)

# ---- Municipal boundary mask (Tudela) -------------------------------------
aoi_poly = gpd.read_file(f"{DATA}/AOI/AOI/AOI.shp")
tudela_mask = rasterize_gdf(aoi_poly) > 0
aoi_mask = ~dem_nodata

print("Water-protection %:", 100*water_mask[aoi_mask].mean())
print("Urban buffer %:", 100*urban_buffer_mask[aoi_mask].mean())
print("Protected-area %:", 100*pna_mask[aoi_mask].mean())
print("NVZ %:", 100*nvz_mask[aoi_mask].mean())
print("Soil-suitable %:", 100*soil_class01[aoi_mask].mean())
print("Land-eligible %:", 100*land_eligible[aoi_mask].mean())

np.savez_compressed(f"{OUT}/covariates.npz",
    dem=dem, slope_pct=slope_pct, soil_class01=soil_class01,
    dist_water=dist_water, water_mask=water_mask,
    dist_urban=dist_urban, urban_buffer_mask=urban_buffer_mask,
    dist_protected=dist_protected, pna_mask=pna_mask,
    nvz_mask=nvz_mask, land_eligible=land_eligible,
    tudela_mask=tudela_mask, aoi_mask=aoi_mask, dem_nodata=dem_nodata,
)

meta = dict(width=width, height=height, pixel=PIXEL,
            transform=list(transform)[:6], crs=str(crs), bounds=list(bounds))
with open(f"{OUT}/grid_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Saved covariates.npz + grid_meta.json")
