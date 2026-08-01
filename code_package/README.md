# Slurry-application suitability analysis — Tudela (Navarre, Spain)
Author: Naziru Halilu

Pure-Python reproduction of a GIS multicriteria workflow for livestock slurry
suitability, run against real geospatial datasets for the municipality of
Tudela (municipal boundary, DEM, lithology, surface water/DPH, urban areas,
protected natural areas, Nitrate Vulnerable Zones, SIGPAC land use/land cover).

No GDAL / GEOS / QGIS / rasterio / Fiona / geopandas / pyproj were available
in the execution environment (and no network access to install them), so
every geoprocessing primitive normally supplied by those libraries (shapefile
parsing, reprojection, polygon rasterisation, buffering, raster algebra,
zonal statistics) was re-implemented from scratch in `code/`.

Figures follow a QGIS-style print-layout composition: a map panel with a
neatline border, a separate bordered legend panel (not overlapping the
mapped data), a north arrow, a scale bar and source/author credits placed in
a reserved margin below the map so nothing overlaps or touches the data.

## Contents
- `code/miniogr.py`        — pure-Python ESRI Shapefile/DBF reader
- `code/geomtools.py`      — polygon rasterisation + Euclidean-distance raster buffering
- `code/muniextract.py`    — single-feature extraction from a large national shapefile
- `code/proj_utm.py`       — ETRS89 geographic -> UTM zone 30N reprojection (Snyder 1987)
- `code/run_analysis.py`   — main analysis pipeline (Steps 1-12)
- `code/make_figures.py`   — QGIS-style cartographic figure generation (Fig. 1-11)
- `figures/`               — all 11 PNG figures (220 dpi) used in the report
- `outputs/stats.json`     — final area statistics (by class and by exclusion criterion)
- `outputs/zonal_stats_TA_PS_parcels.csv` — per-parcel mean suitability score (9,116 parcels)

## Requirements
Python 3 with: numpy, scipy, scikit-image, matplotlib, pillow, pandas.
(No GDAL/GEOS/rasterio/geopandas/QGIS required.)

## Reproducing
1. Point `DATA` in `run_analysis.py` at the extracted contents of the 9 supplied
   zip files (AOI, Digital elevation model, Land use-Land cover, Municipal
   boundary, Nitrate vulnerable and polluted water zones, Protected natural
   areas, Soil characteristics, Surface water bodies, Urban areas).
2. `python run_analysis.py`   → writes out/layers.npz, out/stats.json, out/zonal_stats_TA_PS_parcels.csv
3. `python make_figures.py`   → writes out/figs/fig1..fig11.png
