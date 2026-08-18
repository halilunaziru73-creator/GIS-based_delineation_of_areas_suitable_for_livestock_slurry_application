<div align="center">

# GIS-Based Delineation of Areas Suitable for Livestock Slurry Application

### Case Study — Tudela, Navarre, Spain

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21760026.svg)](https://doi.org/10.5281/zenodo.21760026)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Domain](https://img.shields.io/badge/domain-GIS%20%7C%20Geostatistics%20%7C%20Deep%20Learning-informational)

**Author:** Naziru Halilu

A GIS multicriteria decision analysis (MCDA) suitability overlay, extended with
continuous geostatistical and deep-learning suitability surfaces, for identifying
land suitable for livestock slurry application.

</div>

---


**Workflow sketch**

![Workflow Sketch](workflow_sketch.png)

[View interactive graphical abstract →](https://halilunaziru73-creator.github.io/GIS-based_delineation_of_areas_suitable_for_livestock_slurry_application/)

## Problem, Methodology, and Results

**Problem.** Livestock slurry can substitute mineral fertiliser when applied at the right place, time, and rate, but poses a diffuse pollution risk otherwise. Conventional GIS multicriteria decision analysis (MCDA) delineates suitable land using discrete Boolean/ordinal overlay classes, which discards the continuous, spatially autocorrelated structure of the underlying agro-environmental gradients.

**Methodology.** A conventional seven-criterion GIS-MCDA overlay for Tudela, Navarre (Spain) was extended with two continuous-surface modelling strategies: regression kriging (linear and random-forest trend models) and two deep-learning surrogates (an MLP and a patch-based CNN reading local 9×9 covariate neighbourhoods). All four models were trained on 550 spatially block-cross-validated pseudo-observations drawn from a continuous fuzzy weighted-linear-combination suitability index within the hard-constraint-compliant candidate zone (5.4% of the municipality, 14.35 km²).

**Results.** The random-forest regression-kriging model achieved the best held-out performance (RMSE = 7.78, R² = 0.77), outperforming the deep-learning surrogates (RMSE ≈ 9.9, R² ≈ 0.62) and classical linear-trend kriging (RMSE = 11.0, R² = 0.53), while also producing the lowest residual spatial autocorrelation. Permutation importance identified Nitrate Vulnerable Zone status, slope, and distance to surface water as the dominant covariates.

## Table of Contents

- [Overview](#overview)
- [Methodology Pipeline](#methodology-pipeline)
- [Key Results](#key-results)
- [Figures](#figures)
- [Repository Structure](#repository-structure)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Notes on the Modelling Design](#notes-on-the-modelling-design)
- [Documents](#documents)
- [License](#license)

---

## Overview

This project identifies land suitable for livestock slurry application in the
municipality of Tudela (Navarre, Spain) through two complementary layers of analysis:

| Layer | Method | Output |
|---|---|---|
| **1. Hard-constraint screening** | Rule-based GIS-MCDA overlay (slope, soil, hydrology, land use, protected/nitrate-vulnerable zones) | Binary candidate zone |
| **2. Continuous suitability surface** | Fuzzy Suitability Index (SI), Regression Kriging, and deep-learning surrogates | Continuous 0–100 suitability surface across the candidate zone |

The continuous layer is built and validated with:
- **Regression Kriging** — linear-trend (RK-LM) and random-forest-trend (RK-RF) variants, via PyKrige
- **Deep learning surrogates** — a dense MLP and a patch-based CNN (TensorFlow/Keras)
- **Rigorous validation** — spatial block cross-validation, Moran's I of residuals, permutation feature importance, and a Taylor diagram

---

## Methodology Pipeline

```
 1. build_covariates.py     → 25 m raster covariate stack (slope, distances, soil, land use, NVZ)
 2. mcda_reference.py       → hard-constraint candidate zone + continuous fuzzy Suitability Index
 3. sample_points.py        → pseudo-observation network + spatial block train/test split
 4. regression_kriging.py   → RK-LM & RK-RF fitting, variogram modelling, full-grid prediction
 5. deep_learning.py        → DL-MLP & DL-CNN training, full-grid prediction
 6. validation_stats.py     → Moran's I, Taylor-diagram statistics, permutation importance
 7. make_figures_1–5.py     → all 12 publication-quality figures
```

---

## Key Results

Held-out spatial-block test set, *n* = 127:

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Correlation ↑ | Residual Moran's I ↓ |
|:--|:--:|:--:|:--:|:--:|:--:|
| RK-LM | 11.01 | 8.44 | 0.531 | 0.82 | 0.043 |
| **RK-RF** | **7.78** | **6.53** | **0.765** | **0.89** | **0.017** |
| DL-MLP | 9.92 | 7.49 | 0.619 | 0.83 | 0.038 |
| DL-CNN | 9.90 | 7.82 | 0.620 | 0.87 | 0.061 |

**Random-forest regression kriging (RK-RF)** gave the most accurate and best spatially
behaved reconstruction of the continuous suitability surface — lowest error, highest
fit, and lowest residual spatial autocorrelation (a well-behaved model should leave
little spatial structure in its errors).

---

## Figures

<table>
<tr>
<td width="50%">

**Figure 1 — Study Area & DEM**
![Study area and digital elevation model](figures/fig1_study_area_dem.png)

</td>
<td width="50%">

**Figure 2 — Exclusion Criteria**
![Exclusion criteria layers](figures/fig2_exclusion_criteria.png)

</td>
</tr>
<tr>
<td width="50%">

**Figure 3 — Candidate Zone & Sample Points**
![Candidate zone and sample points](figures/fig3_candidate_zone_samples.png)

</td>
<td width="50%">

**Figure 4 — Reference Suitability Index**
![Reference suitability index](figures/fig4_reference_SI.png)

</td>
</tr>
<tr>
<td width="50%">

**Figure 5 — Variogram**
![Variogram model](figures/fig5_variogram.png)

</td>
<td width="50%">

**Figure 6 — RK Prediction Variance**
![Regression kriging prediction variance](figures/fig6_RK_prediction_variance.png)

</td>
</tr>
<tr>
<td width="50%">

**Figure 7 — Deep Learning Architectures & Training**
![Deep learning architectures and training curves](figures/fig7_DL_architectures_training.png)

</td>
<td width="50%">

**Figure 8 — Observed vs. Predicted**
![Observed vs predicted scatter](figures/fig8_obs_vs_pred_scatter.png)

</td>
</tr>
<tr>
<td width="50%">

**Figure 9 — Taylor Diagram**
![Taylor diagram](figures/fig9_taylor_diagram.png)

</td>
<td width="50%">

**Figure 10 — Model Comparison Maps**
![Model comparison maps](figures/fig10_model_comparison_maps.png)

</td>
</tr>
<tr>
<td width="50%">

**Figure 11 — Ensemble Uncertainty**
![Ensemble uncertainty](figures/fig11_ensemble_uncertainty.png)

</td>
<td width="50%">

**Figure 12 — Feature Importance**
![Permutation feature importance](figures/fig12_feature_importance.png)

</td>
</tr>
</table>

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── Tudela_Slurry_Suitability_Merged_Report.docx      # Combined report: MCDA + RK/DL methodology and results
├── manuscript/
│   └── Slurry_Suitability_RegressionKriging_DeepLearning_Manuscript.docx
│                                                       # Q1-journal-style manuscript — 15 pages, 12 figures,
│                                                       # 2 tables, 14 references, clickable internal citations
├── code/                                              # Full reproducible Python pipeline
│   ├── build_covariates.py
│   ├── mcda_reference.py
│   ├── sample_points.py
│   ├── regression_kriging.py
│   ├── deep_learning.py
│   ├── validation_stats.py
│   ├── make_figures_1.py … make_figures_5.py
│   └── cartohelpers.py                                # shared cartographic helpers (scale bar, north arrow, styling)
├── figures/                                           # All 12 publication-quality figures (300 dpi PNG)
├── outputs_data/                                      # Numeric results (JSON) and pseudo-observation points (CSV)
└── source_data/                                       # Original 8 source GIS layers (see "Source Data" section below)
    ├── AOI/
    ├── Digital elevation model/
    ├── Land use-Land cover/
    ├── Municipal boundary/
    ├── Nitrate vulnerable and polluted water zones/
    ├── Protected natural areas/
    ├── Surface water bodies/
    └── Urban areas/
```

---

## Source Data

The 8 original GIS input layers are included directly in [`source_data/`](./source_data),
so the pipeline can be reproduced without sourcing external data separately:

| Layer | Contents | Format |
|---|---|---|
| `AOI/` | Study-area boundary, DEM subset, lithology, DPH (public hydraulic domain), NVZ clips | Shapefile + GeoTIFF |
| `Digital elevation model/` | MDT25 (25 m) elevation tiles covering the study area | GeoTIFF |
| `Land use-Land cover/` | SIGPAC agricultural parcel polygons + IDUSO land-use code dictionary | Shapefile + CSV/XLSX |
| `Municipal boundary/` | National INSPIRE municipal boundary dataset | Shapefile |
| `Nitrate vulnerable and polluted water zones/` | Designated NVZ boundaries | Shapefile |
| `Protected natural areas/` | Protected natural area polygons | Shapefile |
| `Surface water bodies/` | Public hydraulic domain, wetlands, and buffer-strip polygons | Shapefile |
| `Urban areas/` | IGRPO urban land-registry geopackage | GeoPackage |

> **Note on file size:** `Municipal boundary/…shp` (61.8 MB) and
> `Urban areas/…gpkg` (76.0 MB) exceed GitHub's recommended 50 MB soft limit
> (though both are under the 100 MB hard limit and pushed successfully). If
> you fork this repository and plan to modify these files, consider migrating
> them to [Git LFS](https://git-lfs.github.com/).

---

## How to Run the Code

### 1. Clone the repository

```bash
git clone https://github.com/halilunaziru73-creator/GIS-based_delineation_of_areas_suitable_for_livestock_slurry_application.git
cd GIS-based_delineation_of_areas_suitable_for_livestock_slurry_application
```

### 2. Install dependencies

**Requirements:** `geopandas`, `rasterio`, `shapely`, `pyproj`, `fiona`, `pykrige`,
`scikit-learn`, `tensorflow`, `matplotlib`, `pandas`, `numpy`, `scipy`

```bash
pip install geopandas rasterio shapely pyproj fiona pykrige scikit-learn tensorflow matplotlib pandas numpy scipy
```

### 3. Reproducing the Analysis

The 8 source layers are already included in [`source_data/`](./source_data).
All scripts resolve `source_data/` and `outputs_data/` automatically relative
to the repository root, so they run as-is from any location — no path
editing required.

Run in order from the repository root:

```bash
python code/build_covariates.py      # → covariates.npz, grid_meta.json
python code/mcda_reference.py        # → mcda_reference.npz, mcda_stats.json
python code/sample_points.py         # → sample_points.csv
python code/regression_kriging.py    # → RK_grid_prediction.npz, variogram_residuals.npz, rk_results.json
python code/deep_learning.py         # → DL_grid_predictions.npz, dl_training_history.npz, dl_results.json
python code/validation_stats.py      # → validation_extra.json, feature_importance.json
python code/make_figures_1.py        # ┐
python code/make_figures_2.py        # │
python code/make_figures_3.py        # ├─ figures/*.png
python code/make_figures_4.py        # │
python code/make_figures_5.py        # ┘
```



---

## License

Released under the [MIT License](./LICENSE).

---

## Citation

If you use this repository, please cite it using the metadata in
[`CITATION.cff`](./CITATION.cff) (GitHub renders a "Cite this repository"
button on the repo's main page, in the top-right "About" panel).

## Related work

Part of a broader body of research on GIS, remote sensing, and machine
learning for agronomic and environmental applications:

- [Digital Twin for Gully Biocontrol](https://github.com/halilunaziru73-creator/Digital-Twin-for-the-Evaluation-of-Experimental-Gully-Biocontrol-Using-Morning-Glory-Ipomoea-spp)
- [Geometry-Agnostic Contrastive Learning (GACL)](https://github.com/halilunaziru73-creator/Geometry-Agnostic-Contrastive-Learning-GACL)
- [Real-Time RGB Proxy Vegetation Indexing (N_GACL)](https://github.com/halilunaziru73-creator/Real-Time-RGB-Proxy-Vegetation-Indexing-and-Texture-Analysis-for-UAV-and-Handheld-Crop-Imagery)
- [Hybrid CNN-BiLSTM-Attention for Sediment Transport](https://github.com/halilunaziru73-creator/Hybrid-CNN-BiLSTM-Attention-Sediment-Transport-Agricultural-Gully-System)
- [Operationalizing GIS and ML across Cropping Systems](https://github.com/halilunaziru73-creator/Operationalizing-GIS-and-Machine-Learning-across-Contrasting-Cropping-Systems)
- [Geospatial Data Analysis](https://github.com/halilunaziru73-creator/Geospatial-data-analysis)
