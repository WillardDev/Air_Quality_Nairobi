# Nairobi Air Quality 2025 — PM2.5 Prediction Project

## Problem Statement

Air pollution is a major public health concern in Nairobi, with fine particulate matter (PM2.5) being one of the most harmful pollutants due to its ability to penetrate deep into the lungs and bloodstream. AirQo, a network of low-cost air quality monitoring sensors deployed across Nairobi, collects meteorological and particulate measurements at multiple sites. However, sensor readings frequently contain missing, erroneous, or zero-filled values, and stations have uneven temporal coverage.

Despite the availability of this data, there is no robust model to estimate PM2.5 concentrations at sites where measurements are missing. The goal of this project is to build a **predictive model that estimates daily PM2.5 levels across Nairobi's monitoring sites** using meteorological and spatio-temporal features — enabling better coverage of areas with incomplete data and providing a foundation for air quality forecasting.

## Research Questions

1. What are the spatial and temporal patterns of PM2.5 pollution across Nairobi's monitoring sites?
2. How do temperature and humidity relate to PM2.5 concentrations?
3. Can we accurately predict daily PM2.5 levels using meteorological and spatio-temporal features?
4. Which monitoring sites experience the highest pollution, and what factors drive these differences?
5. How does PM2.5 concentration vary across seasons in Nairobi?

## Project Objectives

1. To analyze the distribution and variation of PM2.5 concentrations across different sites in Nairobi.
2. To explore the relationship between meteorological variables (temperature, humidity) and air quality.
3. To build a predictive regression model for estimating daily PM2.5 levels using available features.
4. To evaluate model performance and identify the most important features influencing air quality predictions.
5. To provide actionable insights for air quality monitoring and public health recommendations.

## Project Overview

This project analyzes a full year (2025) of daily air quality and meteorological readings collected by the AirQo sensor network across 74 sites in Nairobi, Kenya. It follows a full machine learning pipeline:

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
data/air-quality-data-combined.csv
```

## Data Structure

| Column | Type | Description |
| --- | --- | --- |
| `site_name` | object | Name of the monitoring site (e.g., "Athi, Nairobi") |
| `datetime` | datetime | Observation date (daily, UTC), 2025-01-01 to 2025-12-31 |
| `frequency` | object | Aggregation frequency of the reading (`daily`) |
| `network` | object | Sensor network that produced the reading (`airqo`) |
| `latitude` | float | Latitude of the monitoring site |
| `longitude` | float | Longitude of the monitoring site |
| `temperature` | float | Air temperature in °C |
| `humidity` | float | Relative humidity in % |
| `pm2_5` | float | Fine particulate matter concentration (µg/m³) — **target variable** |
| `pm10` | float | Coarse particulate matter concentration (µg/m³) |
| `site_id` | object | Unique identifier for the monitoring site |
| `device_name` | object | Device identifier (e.g., "airqo_g5300") |

**Dataset profile (post-load):**

- **16,374 rows** × **12 columns**, one row per site per day.
- Covers the **full year 2025 (365 days)** across **74 sites**.
- `frequency` is constant (`daily`) and `network` is constant (`airqo`) — both are metadata and dropped from modeling.
- Active sites per day varies significantly, i.e., **coverage is uneven**.

**Target variable (PM2.5):** mean ≈ 22.7 µg/m³, std ≈ 10.7, range 0 – 136 µg/m³.

**Additional variable (PM10):** mean ≈ 30.3 µg/m³, std ≈ 20.2, range 0 – 402 µg/m³ (available for secondary analysis).

**Known data-quality issues (discovered during cleaning):**
- Missing values: latitude/longitude (573), temperature (573), humidity (633), pm2_5 (573), pm10 (574).
- Zero-filled coordinates present (sensor did not report location).
- Zero values in temperature/humidity (sensor not recording weather).
- Rows with pm2_5 = 0.0 (suspected sensor failures).

## Notebook Structure

The notebook (`Air_Quality.ipynb`) follows this exact format — every code block is introduced by a numbered markdown header:

### 1. Data Cleaning
- **1.1** Inspect the DataFrame
- **1.2** Check for missing values
- **1.3** Separate numerical and categorical columns
- **1.4** Impute missing values
- **1.5** Replace zero sensor readings
- **1.6** Check for duplicates
- **1.7** Convert data types
- **1.8** Drop constant columns
- **1.9** Detect outliers
- **1.10** Remove outliers with the IQR method
- **1.11** Verify outliers were removed

### 2. Data Preprocessing
- **2.1** Inspect categorical variables
- **2.2** Split the data into train and test
- **2.3** One-hot encode site_name (train/test separately)
- **2.4** Feature distributions before scaling
- **2.5** Feature scaling

### 3. Exploratory Data Analysis
- **3.0** Summary statistics
- **3.1** Site coverage and sample size per site
- **3.2** Strength/direction of relationships (guides feature selection)
- **3.3** Shape of each distribution / skewness
- **3.4** Relationship between PM2.5 and PM10
- **3.5** Weekday vs Weekend pollution
- **3.6** Which locations are most polluted
- **3.7** Geographic hotspots across Nairobi (choropleth)
- **3.8** Whether heat correlates with pollution levels
- **3.9** Whether humidity correlates with pollution levels
- **3.10** Seasonal patterns: PM2.5 distribution by month
- **3.11** Air Quality Index (AQI) categorization
- **3.12** Handling class imbalance with SMOTE

### 4. Feature Engineering & Selection
- **4.1** Datetime decomposition (+ cyclical encoding)
- **4.2** Site encoding (label encoding)
- **4.3** Lagged & rolling features
- **4.4** Feature selection (correlation with target)

### 5. Model Selection
- **5.1** Data split (regression + classification targets)
- **5.2** Feature scaling
- **5.3** Regression models comparison
  - 5.3.1 Linear Regression
  - 5.3.2 Ridge Regression (+ alpha tuning)
  - 5.3.3 Random Forest Regressor (+ hyperparameter tuning)
  - 5.3.4 Polynomial Regression (degree 2)
  - 5.3.5 Models summary & best model
  - 5.3.6 Comparison chart (RMSE / MAE / R²)
- **5.4** Classification models comparison
  - 5.4.1 Logistic Regression (+ C tuning)
  - 5.4.2 Random Forest Classifier (+ hyperparameter tuning)
  - 5.4.3 Models summary & best model
  - 5.4.4 Accuracy vs Macro-F1 (bar chart)
  - 5.4.5 Per-class precision/recall/F1 (grouped bars)
  - 5.4.6 ROC curves (one-vs-rest)
- **5.5** Classification with SMOTE-balanced training
  - 5.5.1 Class counts before vs after SMOTE (bar chart)

### 6. Error Analysis (winning models only)
- **6.1** Learning curves (train vs cross-validated score by training size)
- **6.2** Regression diagnostics — actual vs predicted scatter, residual plot, residual distribution
- **6.3** Classification diagnostics — confusion matrix, SMOTE vs non-SMOTE comparison
- **6.3.1** False positives & false negatives analysis — per-class FP/FN and decision-threshold tuning to reduce them
- **6.4** Feature importance (both winning Random Forest models)

### 7. Model Explainability
- **7.1** Permutation importance (both winning models)
- **7.2** Partial dependence plots (top features, both winning models)

### 8. Key Findings and Conclusion
- Summary of findings and final remarks.

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