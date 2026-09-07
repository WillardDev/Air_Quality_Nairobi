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

Clean site names, fix zero-filled/missing coordinates, flag implausible sensor readings (pm2_5, temperature, or humidity = 0), resolve duplicate site–day records, and impute remaining missing values using site-specific medians.

### 2. Exploratory Data Analysis (EDA)

Explore PM2.5 distributions, site-to-site pollution differences, monthly seasonality (spikes in Jun–Aug), and correlations between temperature, humidity, and PM2.5.

### 3. Feature Engineering & Feature Selection

Engineer temporal (month, season, day-of-year), spatial (site, coordinates), and meteorological (temperature, humidity, lagged/rolling averages) features, then select the most predictive subset using mutual information and correlation/variance thresholds.

### 4. Model Selection

Compare a gradient of models — baseline mean predictor, ridge/linear regression, and tree ensembles (Random Forest, XGBoost) — to cover interpretable linear baselines through nonlinear ensemble approaches.

### 5. Model Training

Train each candidate model on processed features using time-aware and site-aware splits with 5-fold cross-validation, optimizing for RMSE.

### 6. Model Evaluation & Tuning

Evaluate on held-out data using RMSE, MAE, and R² (with per-site breakdowns) and tune hyperparameters via grid/randomized search to pick the best-performing, stable model.

### 7. Error Analysis

Analyze residuals by time and site to identify systematic under-prediction of pollution spikes and underperforming sites with sparse or missing weather data.

### 8. Model Explainability

Use SHAP and partial dependence plots to rank feature importance, show how site identity and time of year drive predictions, and provide actionable insight.