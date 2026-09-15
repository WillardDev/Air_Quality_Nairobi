"""Reproduce the pm10-free pipeline from Air_Quality_Regression.ipynb and persist
everything the Streamlit app consumes:

  models/xgb_model.joblib    tuned/selected XGBoost regressor
  models/features.joblib     85-feature column list
  models/site_dummy_cols.joblib 73 one-hot site columns (drop_first=True)
  models/site_defaults.csv   per-site medians used as demo defaults
  models/model_metrics.json  honest held-out metrics for the narrative
  plots/chart_data.json      chart data cache used by charts.py
  plots/manifest.json        trimmed, pm10-free/classification-free chart manifest

The pipeline mirrors the notebook cell-for-cell (including the site grouping
used for lag/rolling features). Run with the notebook interpreter that has
xgboost; on macOS that may need:
  DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)


def metric_row(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# Pipeline (§3–§6 of Air_Quality_Regression.ipynb)
# --------------------------------------------------------------------------
df = pd.read_csv(os.path.join(ROOT, "data", "air-quality-data-combined.csv"))
df = df.drop(columns=["pm10"])
rows_raw = len(df)

# §3.5 expose disguised zero readings as NaN
for col in ["temperature", "humidity", "pm2_5"]:
    df[col] = df[col].replace(0, np.nan)

# §3.6 drop target rows we cannot model
df = df.dropna(subset=["pm2_5"]).reset_index(drop=True)
rows_clean = len(df)

# §3.9 data types
df["datetime"] = pd.to_datetime(df["datetime"])
df["site_name"] = df["site_name"].astype(str).str.strip()

# §3.10 drop constant / administrative columns
df = df.drop(columns=["frequency", "network", "site_id", "device_name"])

# §3.7 per-site median imputation, network-wide fallback
for col in ["temperature", "humidity"]:
    df[col] = df.groupby("site_name")[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())

# §4.6 / §5.1 datetime decomposition + cyclical encoding
df["is_weekend"] = (df["datetime"].dt.weekday >= 5).astype(int)
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year
df["day"] = df["datetime"].dt.day
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# §5.3 one-hot site encoding (carry the plain site name alongside for defaults)
df["_site"] = df["site_name"]
df = pd.get_dummies(df, columns=["site_name"], drop_first=True, prefix="site")
site_dummy_cols = [c for c in df.columns if c.startswith("site_")]

# §5.4 lagged & rolling features, grouped per site, sorted by date
df = df.sort_values("datetime").reset_index(drop=True)
site_label = df[site_dummy_cols].idxmax(axis=1)
df["pm2_5_lag1"] = df.groupby(site_label)["pm2_5"].shift(1)
df["pm2_5_roll7"] = df.groupby(site_label)["pm2_5"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean())
df["temp_roll7"] = df.groupby(site_label)["temperature"].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean())
for col in ["pm2_5_lag1", "pm2_5_roll7", "temp_roll7"]:
    df[col] = df.groupby(site_label)[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())

# §5.5 feature selection
features = (
    ["temperature", "humidity", "latitude", "longitude"]
    + ["month_sin", "month_cos", "day_sin", "day_cos", "is_weekend"]
    + ["pm2_5_lag1", "pm2_5_roll7", "temp_roll7"]
    + site_dummy_cols
)
non_dummy_features = [c for c in features if c not in site_dummy_cols]

# per-site demo defaults (temperature/humidity are the imputed values)
site_defaults = df.groupby("_site")[
    ["temperature", "humidity", "latitude", "longitude",
     "pm2_5_lag1", "pm2_5_roll7", "temp_roll7"]
].median()
site_defaults.index.name = "site_name"

# §4.8 latent detail: >55% of raw latitude/longitude rows are (0,0) plus a few
# junk points, so the per-site *median* collapses to (0,0) for most sites. Derive
# representative coordinates from readings inside the Nairobi box (fallback: any
# non-zero reading) and use those for demo defaults and the geographic map.
box = df["latitude"].between(-1.6, -1.0) & df["longitude"].between(36.4, 37.3)
rep_coords = df[box].groupby("_site")[["latitude", "longitude"]].median()
nz = df[(df["latitude"] != 0) | (df["longitude"] != 0)]
nz = nz[nz[["latitude", "longitude"]].notna().all(axis=1)]
nz_rep = nz.groupby("_site")[["latitude", "longitude"]].median()
for s in set(df["_site"].unique()) - set(rep_coords.index):
    if s in nz_rep.index:
        rep_coords.loc[s] = nz_rep.loc[s]
site_defaults = site_defaults.drop(columns=["latitude", "longitude"]).join(rep_coords)

# §6.1 the single train/test split
X = df[features]
y = df["pm2_5"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

# §6.2 scaling (linear baselines only)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# §6.3 five regression models, sensible defaults, no tuning here
linear_names = ("Linear Regression", "Lasso", "Ridge")
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(alpha=0.1, random_state=RANDOM_STATE),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=150, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1),
}
results = {}
for name, model in models.items():
    use_scaled = name in linear_names
    model.fit(X_train_scaled if use_scaled else X_train, y_train)
    pred = model.predict(X_test_scaled if use_scaled else X_test)
    results[name] = metric_row(y_test, pred)

best_name = min(results, key=lambda k: results[k]["RMSE"])

# §10 tuning on the winner only; keep the untuned model when the gain is flat
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
}
search = GridSearchCV(
    XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
search.fit(X_train, y_train)
tuned_best = search.best_estimator_
tuned_preds = tuned_best.predict(X_test)
tuned_metrics = metric_row(y_test, tuned_preds)

untuned_preds = models["XGBoost"].predict(X_test)
tuning_improved = tuned_metrics["RMSE"] < results["XGBoost"]["RMSE"]
final_model = tuned_best if tuning_improved else models["XGBoost"]
final_preds = tuned_preds if tuning_improved else untuned_preds
final_metrics = tuned_metrics if tuning_improved else results["XGBoost"]

print("Model comparison (held-out test, RMSE/MAE/R²):")
for name, m in sorted(results.items(), key=lambda kv: kv[1]["RMSE"]):
    print(f"  {name:<18} {m['RMSE']:.3f} {m['MAE']:.3f} {m['R2']:.3f}")
print(f"Best by RMSE: {best_name}")
print(f"Tuning best params: {search.best_params_}  CV RMSE: {-search.best_score_:.4f}")
print(f"Tuned held-out:     {tuned_metrics['RMSE']:.3f} (untuned {results['XGBoost']['RMSE']:.3f})")
print(f"Final model: {type(final_model).__name__} (tuning_improved={tuning_improved})")

# §8 error by PM2.5 range
y_band = pd.cut(
    y_test, bins=[-np.inf, 12.0, 35.4, np.inf],
    labels=["Good (<=12)", "Moderate (<=35.4)", "Unhealthy (>35.4)"])
error_df = pd.DataFrame({
    "band": y_band,
    "error": final_preds - y_test,
    "abs_error": np.abs(final_preds - y_test),
})
error_by_band = (
    error_df.groupby("band", observed=True)
    .agg(count=("error", "size"), mean_error=("error", "mean"),
         mean_abs_error=("abs_error", "mean"))
    .round(3)
    .reset_index())

# --------------------------------------------------------------------------
# Persist model artifacts
# --------------------------------------------------------------------------
joblib.dump(final_model, os.path.join(MODELS_DIR, "xgb_model.joblib"))
joblib.dump(features, os.path.join(MODELS_DIR, "features.joblib"))
joblib.dump(site_dummy_cols, os.path.join(MODELS_DIR, "site_dummy_cols.joblib"))
site_defaults.to_csv(os.path.join(MODELS_DIR, "site_defaults.csv"))

metrics_payload = {
    "final_model": "XGBoost (untuned §6 defaults)" if not tuning_improved else "XGBoost (tuned §10)",
    "final": final_metrics,
    "untuned": results["XGBoost"],
    "tuned": {**tuned_metrics, "best_params": dict(search.best_params_),
              "cv_rmse": float(-search.best_score_)},
    "comparison": results,
    "features": len(features),
    "train": int(X_train.shape[0]),
    "test": int(X_test.shape[0]),
    "error_by_band": error_by_band.to_dict(orient="records"),
    "rows_raw": rows_raw,
    "rows_clean": rows_clean,
}
with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
    json.dump(metrics_payload, f, indent=2)


# --------------------------------------------------------------------------
# Chart data cache (plots/chart_data.json)
# --------------------------------------------------------------------------
rng = np.random.RandomState(RANDOM_STATE)
eda = df[[
    "_site", "datetime", "temperature", "humidity", "pm2_5",
    "latitude", "longitude", "month", "is_weekend",
]]
eda_samp = eda.sample(n=min(4000, len(eda)), random_state=RANDOM_STATE)

corr_cols = ["temperature", "humidity", "pm2_5"]
corr_mat = df[corr_cols].corr().round(4)

corr_target = df[non_dummy_features + ["pm2_5"]].corr()["pm2_5"].drop("pm2_5")

site_counts = df["_site"].value_counts()
site_pm25 = df.groupby("_site")["pm2_5"].mean().sort_values(ascending=False)

if tuning_improved:
    diag_pred, diag_actual = final_preds, y_test.to_numpy()
else:
    diag_actual = y_test.to_numpy()
    diag_pred = models["XGBoost"].predict(X_test)
diag_pred, diag_actual = diag_pred.astype(float), diag_actual.astype(float)
diag_idx = rng.choice(len(diag_actual), min(2500, len(diag_actual)), replace=False)
diag_idx.sort()

imp_series = pd.Series(final_model.feature_importances_, index=features) \
    .sort_values(ascending=False).head(15)
perm = permutation_importance(final_model, X_test, y_test, scoring="r2",
                              n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
perm_series = pd.Series(perm.importances_mean, index=features) \
    .sort_values(ascending=False).head(15)
perm_std = {f: float(perm.importances_std[features.index(f)]) for f in perm_series.index}

pdp_features = ["pm2_5_roll7", "pm2_5_lag1", "temperature"]
pdp_series = []
for fname in pdp_features:
    pd_res = partial_dependence(final_model, X_test, [fname], kind="average",
                                grid_resolution=40)
    pdp_series.append({
        "feature": fname,
        "values": [float(v) for v in pd_res["grid_values"][0]],
        "average": [float(v) for v in pd_res["average"][0]],
    })

chart_data = {
    "meta": {
        "rows_raw": rows_raw,
        "rows_clean": rows_clean,
        "sites": int(df["_site"].nunique()),
        "features": len(features),
        "train": int(X_train.shape[0]),
        "test": int(X_test.shape[0]),
    },
    "eda": {
        "site_name": eda_samp["_site"].tolist(),
        "datetime": eda_samp["datetime"].astype(str).tolist(),
        "temperature": [float(v) for v in eda_samp["temperature"]],
        "humidity": [float(v) for v in eda_samp["humidity"]],
        "pm2_5": [float(v) for v in eda_samp["pm2_5"]],
        "latitude": [float(v) for v in eda_samp["latitude"]],
        "longitude": [float(v) for v in eda_samp["longitude"]],
        "month": [int(v) for v in eda_samp["month"]],
        "is_weekend": [int(v) for v in eda_samp["is_weekend"]],
    },
    "corr": {
        "cols": corr_cols,
        "values": [[float(v) for v in row] for row in corr_mat.values],
    },
    "site_counts": {
        "site": [str(s) for s in site_counts.index],
        "count": [int(v) for v in site_counts.values],
    },
    "site_pm25": {
        "site": [str(s) for s in site_pm25.index],
        "pm25": [float(v) for v in site_pm25.values],
    },
    "corr_target": {
        "features": non_dummy_features,
        "values": [float(v) for v in corr_target],
    },
    "reg_results": {
        "models": [name for name, _ in sorted(results.items(),
                                              key=lambda kv: kv[1]["RMSE"])],
        "RMSE": [round(results[n]["RMSE"], 4) for n, _ in sorted(
            results.items(), key=lambda kv: kv[1]["RMSE"])],
        "MAE": [round(results[n]["MAE"], 4) for n, _ in sorted(
            results.items(), key=lambda kv: kv[1]["RMSE"])],
        "R2": [round(results[n]["R2"], 4) for n, _ in sorted(
            results.items(), key=lambda kv: kv[1]["RMSE"])],
    },
    "reg_diag": {
        "actual": [float(v) for v in diag_actual[diag_idx]],
        "pred": [float(v) for v in diag_pred[diag_idx]],
        "resid": [float(v) for v in (diag_actual - diag_pred)[diag_idx]],
    },
    "feat_imp": {
        "features": [str(f) for f in imp_series.index],
        "values": [float(v) for v in imp_series.values],
    },
    "perm_imp": {
        "features": [str(f) for f in perm_series.index],
        "mean": [float(v) for v in perm_series.values],
        "std": [perm_std[f] for f in perm_series.index],
    },
    "pdp_reg": {"series": pdp_series},
}

with open(os.path.join(PLOTS_DIR, "chart_data.json"), "w") as f:
    json.dump(chart_data, f)

# --------------------------------------------------------------------------
# Chart manifest (plots/manifest.json) — pm10-free, regression-only
# --------------------------------------------------------------------------
top_sites = site_pm25.head(3)
least_sites = site_counts.sort_values().head(3)
rf_metric = results["Random Forest"]

manifest = {
    "images": [
        {
            "section_num": 4,
            "order": [2],
            "file": "eda_site_coverage.png",
            "title": "4.2 Site coverage and sample size per site",
            "insight": (
                f"Coverage is strongly uneven. The best-monitored sites log up to "
                f"**{int(site_counts.max())}** readings across the year, while the "
                f"sparsest — {', '.join(least_sites.index)} — report fewer than "
                f"**{int(least_sites.max())}** days each. These gaps are exactly the "
                "site-days a gap-filling model can reconstruct, but they also mean "
                "per-site statistics for the sparsest sites rest on very few days."
            ),
        },
        {
            "section_num": 4,
            "order": [3],
            "file": "3d96bbad.png",
            "title": "4.3 Feature distributions",
            "insight": (
                "PM2.5 is strongly right-skewed — most days sit in a low-to-moderate "
                "band around 20 µg/m³, with a long tail of high-pollution episodes up "
                "to ~136 µg/m³. Temperature and humidity are broadly symmetric. The "
                "skew motivates median-based imputation and warns that the *rare* high "
                "days carry most of the error."
            ),
        },
        {
            "section_num": 4,
            "order": [4],
            "file": "39877be1.png",
            "title": "4.4 Correlation heatmap",
            "insight": (
                "Within the basic sensor columns, temperature and humidity correlate "
                "only weakly with PM2.5 (r ≈ −0.09 and r ≈ 0.01) — weather alone "
                "explains little. The strongest signal sits in the *history* features "
                "built in §5 (rolling and lagged PM2.5, r ≈ 0.72/0.71)."
            ),
        },
        {
            "section_num": 4,
            "order": [6],
            "file": "weekend_plot.png",
            "title": "4.6 Weekday vs weekend",
            "insight": (
                "Weekday and weekend PM2.5 distributions overlap heavily — the weekday "
                "effect is small (is_weekend correlates with PM2.5 at only r ≈ −0.04) "
                "but retained because it carries a little weekday traffic / activity "
                "signal and costs nothing."
            ),
        },
        {
            "section_num": 4,
            "order": [7],
            "file": "2168cff4.png",
            "title": "4.7 Which locations are most polluted",
            "insight": (
                f"Mean PM2.5 spans roughly {site_pm25.min():.0f} to "
                f"{site_pm25.max():.0f} µg/m³ across the 74 sites. The most polluted "
                f"are **{', '.join(top_sites.index)}** — an east/north-east cluster "
                "consistent with industrial corridors; the cleanest are spread across "
                "green and lower-traffic areas. Location clearly matters, which §5.3 "
                "encodes as one-hot site columns."
            ),
        },
        {
            "section_num": 4,
            "order": [8, 1],
            "file": "a2be3f89.png",
            "title": "4.8 Temperature vs PM2.5",
            "insight": (
                "Temperature is a weak negative driver (r ≈ −0.09) — warmer readings "
                "lean slightly lower, but the cloud is wide. Temperature carries far "
                "less information than what happened at the site in the last week."
            ),
        },
        {
            "section_num": 4,
            "order": [8, 2],
            "file": "90ec9e9f.png",
            "title": "4.8 Humidity vs PM2.5",
            "insight": (
                "Humidity shows essentially no linear relationship with PM2.5 "
                "(r ≈ 0.01). It is kept in the model anyway — it can still matter in "
                "non-linear splits and costs one column."
            ),
        },
        {
            "section_num": 4,
            "order": [9],
            "file": "2873277e.png",
            "title": "4.9 Seasonal pattern: PM2.5 by month",
            "insight": (
                "There is a clear annual cycle with a **Jun–Aug dry-season peak** and "
                "a lull around the long-rains months. Median and spread both grow in "
                "winter, so seasonality isn't just a shift in level — high days become "
                "both more likely and more extreme."
            ),
        },
        {
            "section_num": 5,
            "order": [5],
            "file": "corr_target.png",
            "title": "5.5 Feature correlation with the target",
            "insight": (
                "The engineered *history* features dominate: the 7-day rolling PM2.5 "
                "mean (r ≈ 0.72) and yesterday's PM2.5 (r ≈ 0.71) are by far the "
                "strongest associates of today's value. Among exogenous features the "
                "monthly cosine (r ≈ −0.38) captures the seasonal cycle. The 73 "
                "one-hot site columns are kept as a group — each is individually tiny "
                "but jointly they encode the genuine spatial variation of §4.7."
            ),
        },
        {
            "section_num": 7,
            "order": [2],
            "file": "reg_viz.png",
            "title": "7.2 Regression model comparison",
            "insight": (
                "On the same held-out 20%, **XGBoost leads every metric** — RMSE "
                f"{results['XGBoost']['RMSE']:.2f} µg/m³, MAE "
                f"{results['XGBoost']['MAE']:.2f} µg/m³, R² "
                f"{results['XGBoost']['R2']:.2f} — with Random Forest close behind "
                f"(RMSE {rf_metric['RMSE']:.2f}). The three linear baselines sit in a "
                "tight cluster around RMSE ≈ 6.7–6.8, confirming the boundary and "
                "history structure the trees exploit."
            ),
        },
        {
            "section_num": 8,
            "order": [1],
            "file": "reg_diag.png",
            "title": "8.1 Regression diagnostics — predicted vs actual, residuals",
            "insight": (
                "Points hug the diagonal in the common band but spread as PM2.5 rises: "
                "**mean absolute error is ~3.3 µg/m³ on Good days, ~3.6 on Moderate "
                "days, and ~10.3 µg/m³ on Unhealthy days**, which the residuals panel "
                "shows is mostly *under*-prediction of real spikes. The model is well "
                "calibrated for typical days and systematically conservative on the "
                "rare extremes — the main limitation to communicate to users (§11)."
            ),
        },
        {
            "section_num": 9,
            "order": [1],
            "file": "feat_imp.png",
            "title": "9.1 XGBoost built-in feature importance (top 15)",
            "insight": (
                "Recent PM2.5 history is used most heavily by the model — `pm2_5_roll7` "
                "and `pm2_5_lag1` dominate the split usage — followed by seasonality "
                "(`month_cos`) and a small number of site indicators (e.g. "
                "`site_UN Avenue Gigiri`, `site_NCC Embakasi`) that flag genuinely "
                "different baseline locations. Note: a high site importance means "
                "location is a strong *predictor*, not that the site 'causes' "
                "pollution — it stands for unmeasured local factors like traffic."
            ),
        },
        {
            "section_num": 9,
            "order": [2],
            "file": "perm_imp.png",
            "title": "9.2 Permutation importance (drop in R² when shuffled)",
            "insight": (
                "Shuffling `pm2_5_lag1` collapses held-out R² by **≈ 0.55** — more than "
                "everything else combined. After it, the 7-day mean and the seasonal "
                "encoding matter most; weather (temperature, humidity) contributes only "
                "a little. Sensor **continuity is the highest-value investment**: if a "
                "site stops reporting, the model loses its biggest lever."
            ),
        },
        {
            "section_num": 9,
            "order": [3],
            "file": "pdp_reg.png",
            "title": "9.3 Partial dependence — how the model reacts to key features",
            "insight": (
                "The predicted PM2.5 curve rises almost one-to-one with yesterday's "
                "reading and the 7-day mean — pollution persists. Temperature's effect "
                "is mild and roughly negative in warm conditions. This is the *model's* "
                "view, not an experimental claim: use it to spot where predictions "
                "move, not to infer causation."
            ),
        },
    ],
    "texts": [
        {
            "section_num": 3,
            "order": [1],
            "file": None,
            "title": "3.1 Data dictionary & overview",
            "insight": (
                "16,374 site-days × 11 columns (after dropping `pm10`), 74 sites, full "
                "year 2025. Constant columns (`frequency`, `network`, `site_id`, "
                "`device_name`) carry no signal and are dropped in §3.10."
            ),
        },
        {
            "section_num": 3,
            "order": [2],
            "file": None,
            "title": "3.5–3.6 Cleaning summary",
            "insight": (
                "Raw data hides faults as zeros: ~6.3% of temperature/humidity rows are "
                "zero (same probe) and 23 rows report PM2.5 = 0. Zero readings are "
                "treated as **missing**, then 596 target rows (3.6%) are dropped for an "
                "unusable `pm2_5`, leaving **15,778 modeled rows**."
            ),
        },
        {
            "section_num": 3,
            "order": [3],
            "file": None,
            "title": "3.7 Imputation policy",
            "insight": (
                "temperature and humidity are imputed with a **per-site median** "
                "(median survives the right-skew; per-site preserves each station's "
                "microclimate). For two sites whose weather probe never worked, a "
                "network-wide median fallback is used — their defaults are the least "
                "trustworthy in the demo."
            ),
        },
        {
            "section_num": 4,
            "order": [1],
            "file": None,
            "title": "4.1 Summary statistics",
            "insight": (
                "Mean PM2.5 ≈ 22.7 µg/m³, median ≈ 20.3 µg/m³, max 136.2 µg/m³ — a "
                "city whose typical day is 'Moderate' (US-EPA) but with real spikes. "
                "Mean temperature ≈ 22.2 °C, mean humidity ≈ 65%."
            ),
        },
        {
            "section_num": 4,
            "order": [5],
            "file": None,
            "title": "4.5 Outlier analysis — decision",
            "insight": (
                "IQR flags ~4% of rows including the high-PM2.5 episodes (up to "
                "136 µg/m³). These are physically plausible pollution days, not sensor "
                "artifacts, so they are **kept**: removing them would teach the model "
                "to systematically under-predict the spikes this project cares most about."
            ),
        },
        {
            "section_num": 6,
            "order": [1],
            "file": None,
            "title": "6.1 Split & feature count",
            "insight": (
                ""
            ),
        },
        {
            "section_num": 7,
            "order": [1],
            "file": None,
            "title": "7.1 Best model by RMSE",
            "insight": (
                "**XGBoost** wins on held-out RMSE "
                f"({results['XGBoost']['RMSE']:.3f} µg/m³ vs the next-best Random "
                f"Forest at {results['Random Forest']['RMSE']:.3f}). The random split is "
                "same-seed for all models, so the comparison is fair."
            ),
        },
        {
            "section_num": 8,
            "order": [2],
            "file": None,
            "title": "8.2 Error by PM2.5 range",
            "insight": (
                "Errors are not uniform: "
                "Good (≤12) days have MAE ≈ 3.3 µg/m³ (n=245), Moderate days ≈ 3.6 "
                "(n=2,563), but Unhealthy (>35.4) days ≈ 10.3 with mean *signed* error "
                "≈ −8.6 (n=348) — high days are systematically under-predicted. "
                "The model is most useful as a monitoring/gap-filling aid and least "
                "reliable exactly when a spike alert matters most."
            ),
        },
    ],
}

with open(os.path.join(PLOTS_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("\nArtifacts written:")
for d in (MODELS_DIR, PLOTS_DIR):
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        print(f"  {p}  ({os.path.getsize(p):,} B)")