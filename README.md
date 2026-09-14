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

The notebook (`Air_Quality.ipynb`) follows the **ML Foundations Capstone 12-section presentation format**. Every code block is introduced by a numbered markdown header, and every figure is followed by a one-line insight.

### 1. Problem Statement
Coverage gaps, passive raw data, and no exposure translation — framed as a **regression** (predict µg/m³) + **classification** (Good / Moderate / Unhealthy) task.

### 2. Business Use Case and Impact
Who needs the tool, the expected impact (cost, timeliness, public health), KPIs, and the deployment path to `app.py`.

### 3. Data Understanding
- **3.1** Data dictionary & overview
- **3.2** Inspect the DataFrame
- **3.3** Check for missing values
- **3.4** Separate numerical and categorical columns
- **3.5** Impute missing values
- **3.6** Replace zero sensor readings
- **3.7** Check for duplicates
- **3.8** Convert data types
- **3.9** Drop constant columns
- **3.10** Detect outliers
- **3.11** Remove outliers with the IQR method
- **3.12** Verify outliers were removed

### 4. Exploratory Data Analysis
- **4.1** Summary statistics
- **4.2** Site coverage and sample size per site
- **4.3** Strength/direction of relationships (guides feature selection)
- **4.4** Shape of each distribution / skewness
- **4.5** Relationship between PM2.5 and PM10
- **4.6** Weekday vs Weekend pollution
- **4.7** Which locations are most polluted
- **4.9** Whether heat correlates with pollution levels
- **4.10** Whether humidity correlates with pollution levels
- **4.11** Seasonal patterns: PM2.5 distribution by month
- **4.12** Air Quality Index (AQI) categorization
  - 4.12.1 Apply categorization to the dataset
- **4.13** Handling class imbalance with SMOTE
  - 4.13.1 Visualize class balance after SMOTE

### 5. Data Preprocessing
- **5.1** Inspect categorical variables
- **5.2** Split the data into train and test
- **5.3** One-hot encode site_name (train/test separately)
- **5.4** Feature distributions before scaling
- **5.5** Feature scaling
- **5.6** Feature engineering
  - 5.6.1 Datetime decomposition (+ 5.6.1.1 cyclical encoding)
  - 5.6.2 Site encoding (label encoding)
  - 5.6.3 Lagged & rolling features
  - 5.6.4 Feature selection

### 6. Model Selection & Modelling
- **6.1** Data split (regression + classification targets)
- **6.2** Feature scaling
- **6.3** Regression models comparison
  - 6.3.1 Linear Regression
  - 6.3.2 Ridge Regression (+ 6.3.2.1 alpha tuning)
  - 6.3.3 Random Forest Regressor (+ 6.3.3.1 tuning)
  - 6.3.4 Polynomial Regression (degree 2)
- **6.4** Classification models comparison
  - 6.4.1 Logistic Regression (+ 6.4.1.1 C tuning)
  - 6.4.2 Random Forest Classifier (+ 6.4.2.1 tuning)
- **6.5** Classification with SMOTE-balanced training

### 7. Model Evaluation
- **7.1** Regression models summary & best model
- **7.2** Regression model comparison chart (RMSE / MAE / R²)
- **7.3** Classification models summary & best model
- **7.4** Accuracy vs macro-F1 (bar chart)
- **7.5** Per-class precision/recall/F1 (grouped bars)
- **7.6** ROC curves (one-vs-rest)
- **7.7** Class counts before vs after SMOTE (bar chart)
- **7.8** Confusion matrices (winning & SMOTE classifiers)

### 8. Error Analysis
- **8.1** Learning curves (train vs cross-validated score by training size)
- **8.2** Regression diagnostics — actual vs predicted scatter, residuals
- **8.3** False positives & false negatives analysis — per-class FP/FN, plus decision-threshold tuning to reduce them

### 9. Model Explainability
- **9.1** Feature importance (both winning Random Forest models)
- **9.2** Permutation importance (both winning models)
- **9.3** Partial dependence plots (top features)

### 10. Hyperparameter Tuning
GridSearchCV approach and a summary of every tuned model: search spaces, champion parameters, and held-out metrics.

### 11. Final Model & Recommendations
Champion models, recommendations for deployment and operations, key findings, and the conclusion.

### 12. Project Demonstration
Exporting the winning models, scaler, encoders, feature list, and per-site defaults to `models/` so the Streamlit app can serve predictions.

## Streamlit UI

An interactive web interface (`app.py`) lets you predict PM2.5 and its category without touching the notebook.

```
pip install -r requirements.txt     # (or: pip install streamlit plotly joblib sklearn pandas numpy)
streamlit run app.py
```

The app mirrors the notebook's numbered sections as tabs:
- **1. Problem Statement** — project title, problem statement, objectives, research questions, and the pipeline at a glance.
- **2. Business Use Case & Impact** — stakeholders, expected impact, KPIs, and the deployment path.
- **3. Data Understanding** — dataset metrics (74 sites, 16,374 rows) plus the cleaning decisions and figures.
- **4. Exploratory Data Analysis** — key insights plus all 13 EDA figures.
- **5. Data Preprocessing** — the 14 engineered features grouped by weather/location, calendar, and history, plus the preprocessing/feature-engineering figures.
- **6. Model Selection & Modelling** — regression vs classification comparison, why Random Forest won, and the comparison/tuning figures.
- **7. Model Evaluation** — winner summary cards plus the summary, ROC, SMOTE, and confusion-matrix figures.
- **8. Error Analysis** — learning curves, regression diagnostics, and FP/FN analysis figures.
- **9. Model Explainability** — permutation importance and partial dependence figures.
- **10. Hyperparameter Tuning** — a GridSearchCV summary table with the champion parameters for every tuned model.
- **11. Final Model & Recommendations** — champion models, key findings, conclusion, and operational recommendations.
- **12. Project Demonstration** — **interactive predictor**: pick a **site** and adjust **temperature, humidity, and PM10** on the page (not a sidebar). Recent 7-day readings and location come from that site's typical values; the app shows the predicted PM2.5 concentration, the predicted category with class probabilities, and a colored gauge showing where the value falls on the US-EPA breakpoints (Good ≤ 12.0, Moderate ≤ 35.4, Unhealthy > 35.4 µg/m³).

Every figure comes with its one-line insight directly beneath it, exactly as written in the notebook.

The app loads pre-trained artifacts from `models/` (generated by notebook §12). Re-run that cell after retraining to refresh the UI.

### Model artifacts (`models/`)
| File | Contents |
| --- | --- |
| `rf_reg.joblib` | Tuned Random Forest regressor (predicts PM2.5) |
| `rf_clf.joblib` | Tuned Random Forest classifier (predicts category) |
| `scaler.joblib` | StandardScaler fitted on training features |
| `label_encoder.joblib` | Good/Moderate/Unhealthy → 0/1/2 |
| `site_encoder.joblib` | 74 site names → numeric codes |
| `features.joblib` | Feature list (model input order) |
| `site_defaults.csv` | Per-site defaults to pre-fill the UI sliders |

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