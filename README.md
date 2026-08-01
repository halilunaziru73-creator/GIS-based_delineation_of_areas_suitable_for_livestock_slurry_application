# GIS-based Delineation of Areas Suitable for Livestock Slurry Application

**Case study: Tudela, Navarre, Spain**
Author: Naziru Halilu

A pure-Python reproduction of a GIS multicriteria decision analysis (MCDA) workflow
for identifying land suitable for livestock slurry application, applied to real
geospatial datasets for the municipality of Tudela (municipal boundary, DEM,
lithology, surface water / DPH, urban areas, protected natural areas, Nitrate
Vulnerable Zones, and SIGPAC land use/land cover).

No GDAL, GEOS, QGIS, rasterio, Fiona, geopandas, or pyproj were available in the
execution environment (and no network access to install them), so every
geoprocessing primitive normally supplied by those libraries — shapefile parsing,
reprojection, polygon rasterisation, buffering, raster algebra, and zonal
statistics — was re-implemented from scratch.

Output figures follow a QGIS-style print-layout composition: a map panel with a
neatline border, a separate bordered legend panel, a north arrow, a scale bar,
and source/author credits placed in a reserved margin so nothing overlaps the
mapped data.

## Repository structure

```
.
├── Slurry_Suitability_Tudela_Report.docx   # Full written report
└── code_package/
    ├── README.md                           # Detailed code documentation
    ├── code/
    │   ├── miniogr.py         # Pure-Python ESRI Shapefile/DBF reader
    │   ├── geomtools.py       # Polygon rasterisation + distance-raster buffering
    │   ├── muniextract.py     # Single-feature extraction from a national shapefile
    │   ├── proj_utm.py        # ETRS89 geographic → UTM zone 30N reprojection
    │   ├── run_analysis.py    # Main analysis pipeline (Steps 1–12)
    │   └── make_figures.py    # QGIS-style cartographic figure generation
    ├── figures/                            # 11 output figures (220 dpi PNG)
    └── outputs/
        ├── stats.json                      # Final area statistics
        └── zonal_stats_TA_PS_parcels.csv   # Per-parcel suitability scores (9,116 parcels)
```

## Requirements

- Python 3
- numpy, scipy, scikit-image, matplotlib, pillow, pandas

No GDAL / GEOS / rasterio / geopandas / QGIS required.

## Reproducing the analysis

1. Point `DATA` in `code_package/code/run_analysis.py` at the extracted contents
   of the supplied source datasets (AOI, DEM, land use/land cover, municipal
   boundary, nitrate vulnerable and polluted water zones, protected natural
   areas, soil characteristics, surface water bodies, urban areas).
2. Run the analysis pipeline:
   ```bash
   python code_package/code/run_analysis.py
   ```
   This writes `layers.npz`, `stats.json`, and `zonal_stats_TA_PS_parcels.csv`.
3. Generate the figures:
   ```bash
   python code_package/code/make_figures.py
   ```
   This writes `fig1.png` through `fig11.png`.

## Report

The full methodology, criteria weighting, and results discussion are documented
in [`Slurry_Suitability_Tudela_Report.docx`](./Slurry_Suitability_Tudela_Report.docx).

## License

No license has been specified yet — all rights reserved by default. Add a
`LICENSE` file if you'd like to permit reuse.
