# Nairobi Air Quality 2025 — PM2.5 Prediction Project

## Problem Statement

Air pollution is a major public health concern in Nairobi, with fine particulate matter (PM2.5) being one of the most harmful pollutants due to its ability to penetrate deep into the lungs and bloodstream. AirQo, a network of low-cost air quality monitoring sensors deployed across Nairobi, collects meteorological and particulate measurements at multiple sites. However, sensor readings frequently contain missing, erroneous, or zero-filled values, and stations have uneven temporal coverage.

Despite the availability of this data, there is no robust model to estimate PM2.5 concentrations at sites where measurements are missing. The goal of this project is to build a **predictive model that estimates daily PM2.5 levels across Nairobi's monitoring sites** using meteorological and spatio-temporal features — enabling better coverage of areas with incomplete data and providing a foundation for air quality forecasting.

## Project Overview

This project analyzes a full year (2025) of daily air quality and meteorological readings collected by the AirQo sensor network across 10 sites in Nairobi, Kenya. It follows a full machine learning pipeline:

1. **Data ingestion** — load the raw CSV from `data/`.
2. **Data cleaning & preprocessing** — handle missing values, zero-filled sensor readings, coordinate inconsistencies, and incomplete temporal coverage.
3. **Exploratory Data Analysis (EDA)** — understand distributions, site-to-site variation, seasonal patterns, and correlations between temperature, humidity, and PM2.5.
4. **Feature engineering & selection** — derive temporal features (day of year, season, month), site-level features, and lagged/rolling meteorological features to maximize predictive signal.
5. **Modeling** — train and compare candidate regression models to predict PM2.5.
6. **Evaluation, tuning, error analysis & explainability** — rigorously validate, tune, diagnose failure modes, and interpret model decisions.

The output of the pipeline is a validated PM2.5 prediction model plus supporting visualizations (saved to `plots/`).

## Tools

| Tool | Use |
| --- | --- |
| **Python 3** | Primary programming language |
| **pandas / NumPy** | Data loading, cleaning, manipulation |
| **Matplotlib / Seaborn** | Exploratory data analysis and visualization |
| **scikit-learn / XGBoost** | Feature selection, model training, and evaluation |
| **SHAP** | Model explainability |

### Data Source

The dataset comes from the **AirQo** low-cost air quality monitoring network in Nairobi, Kenya. It is stored as a single CSV file:

```
data/nairobi_air_quality_2025.csv
```

## Data Structure

| Column | Type | Description |
| --- | --- | --- |
| `site_name` | object | Name of the monitoring site (e.g., "Fire Station, Nairobi") |
| `datetime` | datetime | Observation date (daily, UTC), 2025-01-01 to 2025-12-31 |
| `frequency` | object | Aggregation frequency of the reading (`daily`) |
| `network` | object | Sensor network that produced the reading (`airqo`) |
| `latitude` | float | Latitude of the monitoring site |
| `longitude` | float | Longitude of the monitoring site |
| `temperature` | float | Air temperature in °C |
| `humidity` | float | Relative humidity in % |
| `pm2_5` | float | Fine particulate matter concentration (µg/m³) — **target variable** |

**Dataset profile (post-load):**

- **2,799 rows** × **9 columns**, one row per site per day.
- Covers the **full year 2025 (365 days)** across **10 sites**.
- `frequency` is constant (`daily`) and `network` is constant (`airqo`) — both are metadata and dropped from modeling.
- Active sites per day ranges from **4 to 10** (mean ≈ 7.7), i.e., **coverage is uneven**.

**Monitoring sites & sample size:**

| Site | Records |
| --- | --- |
| Fire Station, Nairobi | 611 |
| Athi, Nairobi | 365 |
| Buruburu, Makadara, Nairobi | 365 |
| Donholm, Embakasi East, Nairobi | 334 |
| Komarock Nairobi, Kenya | 334 |
| Kuwinda Lang'ata Road | 235 |
| Birongo Square, Nairobi West | 184 |
| Drumvale Drive, Kamulu | 184 |
| Kiamaiko | 172 |
| Kenyatta University, Nairobi | 15 |

**Target variable (PM2.5):** mean ≈ 23.5 µg/m³, std ≈ 10.8, range 0 – 120 µg/m³. `Fire Station` is the most polluted site (mean ≈ 29.1 µg/m³) and `Athi` / `Drumvale Drive` the cleanest (mean ≈ 19.7 µg/m³).

**Known data-quality issues (discovered during cleaning):**
- Missing values: latitude/longitude (43), temperature (45), humidity (102), pm2_5 (43).
- ~1,126 rows (40%) have zero-filled coordinates (sensor did not report location).
- `Buruburu` has temperature/humidity effectively zero across the year (sensor not recording weather).
- 6 rows with pm2_5 = 0.0 (suspected sensor failures).
- 317 duplicate site–day pairs (multiple readings for the same site on the same day) and/or sites reporting more than 365 records, indicating inconsistent ingestion.

## Methodology

### 1. Data Cleaning & Preprocessing

- Parse `datetime` to a proper datetime type; extract UTC date and calendar features.
- Drop metadata columns that carry no information (`frequency`, `network`).
- Standardize site names into a clean `site` label and correct site coordinates using per-site representative (non-zero) latitude/longitude where coordinates are zero-filled or missing.
- Remove or flag implausible sensor readings: pm2_5 = 0, temperature = 0, humidity = 0 for sites that otherwise report valid weather (Buruburu is treated as having missing weather, not zero weather).
- Resolve/inspect the 317 duplicate site–day records and enforce one row per site per day.
- Handle missing values via **site-specific imputation** (per-site median) or drop rows where the target `pm2_5` is missing.
- Restrict analysis to sites with sufficient coverage (drop `Kenyatta University` if its 15 records prove non-representative).
- Verify no target leakage: all features used at prediction time must be observable without the pm2_5 value.

### 2. Exploratory Data Analysis (EDA)

- **Distribution analysis:** histograms and summary statistics for pm2_5, temperature, and humidity (right-skewed PM2.5 with extreme spikes at Fire Station and Kuwinda).
- **Site comparisons:** boxplots of pm2_5 by site to highlight the most vs. least polluted areas.
- **Temporal analysis:** daily and monthly mean pm2_5 to reveal seasonality — higher concentrations around **June–August** (mean ≈ 30+ µg/m³) and lower around **November–December** (mean ≈ 15 µg/m³).
- **Correlation analysis:** temperature and humidity are strongly correlated with each other (r ≈ 0.82) but only weakly with pm2_5 (r ≈ 0.05 and 0.04), signaling that nonlinear and spatio-temporal features will be needed.
- **Missingness analysis:** map of missing/zero values across sites to decide the imputation strategy.

### 3. Feature Engineering & Feature Selection

**Engineered features:**
- **Temporal:** month, day of year, season (long/short rain, dry), weekend vs. weekday.
- **Spatial:** site identifier, site mean PM2.5, site latitude/longitude.
- **Meteorological:** raw temperature and humidity, plus lagged (t-1, t-7) and rolling (7- and 30-day) averages.
- **Interaction terms:** temperature × humidity, and month × site for local seasonal profiles.

**Selection:**
- Use **mutual information** and **correlation thresholds** to drop redundant features (e.g., one of temperature/humidity given collinearity).
- Prune near-constant features (low variance threshold).
- Validate selected features by comparing RMSE with and without each candidate group (backward selection).

### 4. Model Selection

Candidate regression models, selected to cover linear to nonlinear/ensemble approaches:

1. **Baseline:** mean predictor (site-specific mean PM2.5).
2. **Linear:** Ridge Regression / Linear Regression (interpretable, tests linearity).
3. **Tree ensembles:** Random Forest Regressor and **XGBoost** (capture non-linearities and interactions, strong tabular performance).
4. **Optional:** LightGBM for comparison if runtime permits.

Models are chosen based on prior evidence that tree ensembles dominate for small-meterological-tabular datasets, while linear models provide the interpretable baseline.

### 5. Model Training

- **Split strategy:** site-aware and time-aware splits (e.g., train on Jan–Sep, validate Oct–Nov, test Dec) to avoid leakage and simulate forecasting; optionally `GroupKFold` by site.
- **Preprocessing pipeline:** one-hot encoding of site and season, scaling of numeric features.
- **Cross-validation:** 5-fold CV for hyperparameter search.
- **Loss:** RMSE (primary), MAE and R² reported for interpretability.
- Train each candidate model on the tuned feature set and record CV performance before fine-tuning.

### 6. Model Evaluation & Tuning

- **Metrics:** RMSE, MAE, R² and MAPE on the held-out test set; report per-site errors to check generalization.
- **Tuning:** `GridSearchCV`/`RandomizedSearchCV` (or Optuna) for XGBoost/RandomForest — n_estimators, max_depth, learning_rate, subsample, colsample_bytree.
- **Model comparison:** prefer the model with the best **test RMSE** at a similar or lower variance across folds; guard against overfitting by tracking train–test gap.
- **Robustness checks:** residuals vs. time and site; verify performance on sites with sparse data.

### 7. Error Analysis

- Plot **residuals vs. predicted** and **residuals vs. month/site** to find where the model under- or over-predicts.
- Identify failure modes: high-pollution spikes (e.g., > 100 µg/m³) are likely systematically under-predicted since they are rare and extreme.
- Sites with very few records (e.g., Kenyatta University) will show unstable predictions — quantify and report per-site uncertainty.
- Quantify performance when weather sensors fail (e.g., Buruburu) to check reliance on temperature/humidity.
- Iterate on features/cleaning choices if a systematic pattern appears.

### 8. Model Explainability

- **SHAP analysis** on the final model (XGBoost/RandomForest) to rank global feature importance and direction of effect.
- **Per-site SHAP** to show how time-of-year, temperature, and site identity drive predictions in different neighborhoods.
- **Partial dependence plots** for the strongest features (e.g., month, site mean PM2.5).
- Summarize actionable insights: which sites have systematic over/under-prediction, which features dominate, and whether weather data actually helps or whether site/calendar features alone suffice.