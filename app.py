import json as _json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

CATEGORY_COLORS = {"Good": "#2e8b57", "Moderate": "#d4a017", "Unhealthy": "#b22222"}
CATEGORY_ORDER = ["Good", "Moderate", "Unhealthy"]


@st.cache_resource
def load_artifacts():
    return {
        "rf_reg": joblib.load(os.path.join(MODELS_DIR, "rf_reg.joblib")),
        "rf_clf": joblib.load(os.path.join(MODELS_DIR, "rf_clf.joblib")),
        "scaler": joblib.load(os.path.join(MODELS_DIR, "scaler.joblib")),
        "label_encoder": joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib")),
        "site_encoder": joblib.load(os.path.join(MODELS_DIR, "site_encoder.joblib")),
        "features": joblib.load(os.path.join(MODELS_DIR, "features.joblib")),
        "site_defaults": pd.read_csv(
            os.path.join(MODELS_DIR, "site_defaults.csv")
        ).set_index("site_name"),
    }


@st.cache_resource
def load_viz_manifest():
    path = os.path.join(PLOTS_DIR, "manifest.json")
    if not os.path.exists(path):
        return []
    return _json.load(open(path))


def build_feature_vector(site_encoded, datetime, temp, humidity, pm10, lat, lon,
                         pm2_5_lag1, pm2_5_roll7, temp_roll7):
    month = datetime.month
    day_of_week = datetime.weekday()
    return [
        temp, humidity, pm10, lat, lon, site_encoded,
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * day_of_week / 7), np.cos(2 * np.pi * day_of_week / 7),
        int(day_of_week >= 5),
        pm2_5_lag1, pm2_5_roll7, temp_roll7,
    ]


def category_gauge(pm2_5_pred):
    boundaries = [(12.0, "Good", "#2e8b57"), (35.4, "Moderate", "#d4a017")]
    limit = 55.0
    fig = go.Figure()
    start = 0.0
    for upper, _, color in boundaries:
        fig.add_shape(type="rect", x0=start, x1=min(upper, limit), y0=0, y1=0.35,
                      fillcolor=color, opacity=0.6, line_width=0)
        start = upper
    fig.add_shape(type="rect", x0=start, x1=limit, y0=0, y1=0.35,
                  fillcolor="#b22222", opacity=0.6, line_width=0)
    clipped = min(pm2_5_pred, limit)
    fig.add_trace(go.Scatter(x=[clipped], y=[0.175], mode="markers+text",
                             marker=dict(size=14, color="#0f1115"),
                             text=[f"{pm2_5_pred:.1f}"], textposition="top center",
                             showlegend=False))
    for x, label in [(12.0, "Good | 12"), (35.4, "Moderate | 35.4"), (55.0, "55+")]:
        fig.add_annotation(x=x, y=-0.12, text=label, showarrow=False, font=dict(size=10))
    fig.update_xaxes(range=[0, limit], showgrid=False, visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=40),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def probability_bars(classes, proba):
    fig = go.Figure(go.Bar(
        x=classes, y=list(proba),
        marker_color=[CATEGORY_COLORS[c] for c in classes],
        text=[f"{p * 100:.1f}%" for p in proba], textposition="outside",
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Probability", yaxis=dict(range=[0, 1.15]),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def render_section_viz(section_number):
    """Show every notebook figure belonging to top-level section `section_number`."""
    manifest = load_viz_manifest()
    if not manifest:
        st.warning("No `plots/manifest.json` found — visualizations are unavailable.")
        return
    images = manifest.get("images", []) if isinstance(manifest, dict) else manifest
    entries = [v for v in images if v["section"].startswith(f"{section_number}. ")]
    if not entries:
        st.info(f"No saved figures for this section.")
        return
    for v in entries:
        st.markdown(f"##### {v['title']}")
        st.image(os.path.join(PLOTS_DIR, v["file"]))
        if v.get("insight"):
            st.markdown(f"💡 {v['insight']}")
        st.divider()


def render_section_insights(section_number):
    """Show the notebook's textual (figures-independent) insights for this section."""
    manifest = load_viz_manifest()
    if not manifest:
        return
    texts = manifest.get("texts", []) if isinstance(manifest, dict) else []
    entries = [v for v in texts if v["section"].startswith(f"{section_number}. ")]
    if not entries:
        return
    with st.expander(f"📝 Key insights from this section ({len(entries)})"):
        for v in entries:
            st.markdown(f"- **{v['title']}** — {v['insight']}")


st.set_page_config(page_title="Nairobi Air Quality — PM2.5 Prediction", layout="wide")

artifacts = load_artifacts()
features = artifacts["features"]
site_defaults = artifacts["site_defaults"]
site_encoder = artifacts["site_encoder"]
classes = list(artifacts["label_encoder"].classes_)

tabs = st.tabs(
    ["Project Overview",
     "1. Data Cleaning & Preprocessing",
     "2. Exploratory Data Analysis",
     "3. Feature Engineering & Selection",
     "4. Model Selection",
     "5. Error Analysis",
     "6. Model Explainability",
     "7. Deployment — Interactive Predictor",
     "8. Key Findings & Conclusion",
    ]
)

# ---------------------------------------------------------------------------
# Tab 0 — Project Overview (problem, objectives, research questions)
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.title("Nairobi Air Quality — PM2.5 Prediction")
        st.caption("A full machine learning pipeline on AirQo sensor data (2025)")

    st.markdown(
        "### Project overview\n"
        "This project analyzes a full year (2025) of daily air quality and "
        "meteorological readings collected by the AirQo sensor network across "
        "**74 sites** in Nairobi, Kenya. It follows a full machine learning pipeline: "
        "ingestion, cleaning, exploratory analysis, feature engineering, model "
        "selection (regression + classification), error analysis, explainability, "
        "and deployment — concluding with this interactive Streamlit app."
    )

    st.markdown("### Problem statement")
    st.markdown(
        "Air pollution is a major public health concern in Nairobi, with fine "
        "particulate matter (PM2.5) being one of the most harmful pollutants due to "
        "its ability to penetrate deep into the lungs. AirQo's low-cost sensor "
        "network spans the city, but readings frequently contain **missing, "
        "erroneous, or zero-filled values**, and stations have uneven temporal "
        "coverage. There is no robust model to estimate PM2.5 at sites where "
        "measurements are missing — leaving monitoring gaps."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("### Project objectives")
        for o in [
            "Analyze the distribution and variation of PM2.5 across Nairobi's sites.",
            "Explore how temperature and humidity relate to air quality.",
            "Build a predictive regression model for daily PM2.5 from available features.",
            "Evaluate performance and identify the most important predictive features.",
            "Provide actionable insight for air quality monitoring and public health.",
        ]:
            st.markdown(f"- {o}")
    with right:
        st.markdown("### Research questions")
        for q in [
            "What spatial and temporal patterns shape PM2.5 across Nairobi?",
            "How do temperature and humidity relate to PM2.5?",
            "Can we accurately predict daily PM2.5 from these features?",
            "Which sites are most polluted and why?",
            "How does PM2.5 vary across seasons in Nairobi?",
        ]:
            st.markdown(f"- {q}")

    st.markdown("### Pipeline at a glance")
    steps = [
        ("🧹 Data Cleaning", "missing values, zeros, outliers"),
        ("🧊 Preprocessing", "imputation, scaling, encoding"),
        ("🔍 EDA", "seasonality, site variation"),
        ("🧮 Feature Engineering", "calendar, site, lagged/rolling"),
        ("⚖️ Model Selection", "regression + classification"),
        ("🩺 Error Analysis", "diagnostics on winning models"),
        ("💡 Explainability", "importance, partial dependence"),
        ("🚀 Deployment", "this Streamlit app"),
    ]
    for step, desc in steps:
        st.markdown(f"**{step}** — {desc}")

# ---------------------------------------------------------------------------
# Tab 1 — Data Cleaning & Preprocessing
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("## 1. Data Cleaning & Preprocessing")
    st.markdown(
        "One year (2025) of **daily** readings: **16,374 rows × 12 columns** across "
        "**74 sites**, covering site name, datetime, coordinates, temperature, "
        "humidity, PM2.5, and PM10."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sites", "74")
    c2.metric("Rows (raw)", "16,374")
    c3.metric("Coverage", "Full year 2025")
    c4.metric("Rows (modeled)", "15,179")

    st.markdown("### What was cleaned")
    for item in [
        "Dropped constant columns (`frequency`, `network`, `site_id`, `device_name`)",
        "Imputed missing values with **(site-specific) median**",
        "Replaced zero-valued temperature/humidity (sensor glitches) with imputed medians",
        "Removed outliers via **IQR bounds** after inspecting distributions",
        "Resolved duplicated `latitude`/`longitude` columns",
    ]:
        st.markdown(f"- {item}")

    st.markdown("### Before / after")
    render_section_viz(1)
    render_section_viz(2)
    render_section_insights(1)
    render_section_insights(2)

# ---------------------------------------------------------------------------
# Tab 2 — Exploratory Data Analysis
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("## 2. Exploratory Data Analysis")
    st.markdown(
        "Distribution shapes, site-by-site variation, seasonal patterns, and "
        "relationships between temperature, humidity, PM10, and PM2.5."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("- Typical reading ≈ **22.7 µg/m³** (Moderate range)")
        st.markdown("- Clear **Jun–Aug pollution peak** (monthly trend)")
        st.markdown("- **PM10** correlates strongly with PM2.5 (r ≈ 0.97)")
    with right:
        st.markdown("- Coverage is uneven — some sites log far more days than others")
        st.markdown("- More observations on weekdays than weekends")
        st.markdown("- Class imbalance: 87% of days are `Moderate`")

    render_section_viz(3)
    render_section_insights(3)

# ---------------------------------------------------------------------------
# Tab 3 — Feature Engineering & Selection
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("## 3. Feature Engineering & Selection")
    st.markdown(
        "From raw sensor readings we engineer **14 features** across four groups: "
        "weather, location, calendar, and history."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Weather & location")
        for f in ["temperature", "humidity", "pm10", "latitude", "longitude",
                  "site_encoded"]:
            st.markdown(f"- `{f}`")
    with c2:
        st.subheader("Calendar")
        for f in ["month_sin", "month_cos", "day_sin", "day_cos", "is_weekend"]:
            st.markdown(f"- `{f}`")
    with c3:
        st.subheader("History (lagged/rolling)")
        for f in ["pm2_5_lag1", "pm2_5_roll7", "temp_roll7"]:
            st.markdown(f"- `{f}`")

    st.markdown(
        "Cyclical sin/cos encoding keeps month and day-of-week distances meaningful; "
        "1-day and 7-day lagged/rolling PM2.5 plus rolling temperature capture "
        "recent history. `site_encoded` (LabelEncoder) carries site identity."
    )
    st.markdown("### What drives predictions?")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Feature importance** (both models):\n"
            "- `pm10` (dominant)\n"
            "- `pm2_5_roll7` (7-day mean)\n"
            "- `pm2_5_lag1` (yesterday)\n\n"
            "Recent readings carry most of the signal."
        )
    with right:
        st.markdown(
            "**Seasonality** is the strongest exogenous driver — monthly sin/cos "
            "encoding plus temperature capture the Jun–Aug peak.\n\n"
            "**Space matters less** once meteorology and history are known."
        )

    render_section_insights(4)

# ---------------------------------------------------------------------------
# Tab 4 — Model Selection
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("## 4. Model Selection")
    st.markdown(
        "Both tasks run on the same 14 engineered features, an **80/20 stratified "
        "split**, standardized inputs, and the same random seed for fair comparison."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Regression (predict PM2.5)")
        st.markdown(
            "**Tuned Random Forest** wins: RMSE ≈ 1.97, MAE ≈ 1.32, **R² ≈ 0.95**.\n\n"
            "Also compared: Linear, Ridge, Polynomial (degree 2), and default Random Forest."
        )
    with c2:
        st.subheader("Classification (Good/Moderate/Unhealthy)")
        st.markdown(
            "**Tuned Random Forest** wins: **≈ 92% accuracy**, macro-F1 ≈ 0.84.\n\n"
            "Also compared: Logistic Regression, and SMOTE-balanced Random Forest "
            "(which improved recall on the rare classes)."
        )

    st.markdown("### Why Random Forest?")
    for item in [
        "Captures non-linear, seasonal, site-specific structure",
        "Robust to the missing/sparse data of low-cost sensor networks",
        "Feature-importance and partial-dependence insights are model-agnostic-friendly",
        "Scale-invariant — safe to serve from exported artifacts",
    ]:
        st.markdown(f"- {item}")

    render_section_viz(5)
    render_section_insights(5)

# ---------------------------------------------------------------------------
# Tab 5 — Error Analysis
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("## 5. Error Analysis")
    st.markdown(
        "A deep-dive into the **winning Random Forest models**: learning curves, "
        "regression residuals, confusion matrices (with SMOTE comparison and "
        "threshold tuning), and feature importance."
    )
    render_section_viz(6)
    render_section_insights(6)

# ---------------------------------------------------------------------------
# Tab 6 — Model Explainability
# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown("## 6. Model Explainability")
    st.markdown(
        "Model-agnostic tools — permutation importance and partial dependence "
        "plots — explain *why* the winning models make their predictions."
    )
    render_section_viz(7)

# ---------------------------------------------------------------------------
# Tab 7 — Deployment: Interactive Predictor
# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("## 7. Deployment — Interactive Predictor")
    st.caption("Adjust the four key parameters and watch the predictions update.")

    col_site, col_temp, col_hum, col_pm10 = st.columns(4)
    with col_site:
        site = st.selectbox("Site", site_defaults.index, index=0)
    defaults = site_defaults.loc[site]
    with col_temp:
        temperature = st.number_input(
            "Temperature (°C)", min_value=-5.0, max_value=40.0,
            value=round(float(defaults["temperature"]), 1), step=0.1,
        )
    with col_hum:
        humidity = st.number_input(
            "Humidity (%)", min_value=0.0, max_value=100.0,
            value=round(float(defaults["humidity"]), 1), step=0.1,
        )
    with col_pm10:
        pm10 = st.number_input(
            "PM10 (µg/m³)", min_value=0.0, max_value=500.0,
            value=round(float(defaults["pm10"]), 1), step=0.1,
        )

    site_encoded_val = int(site_encoder.transform([site])[0])
    vec = build_feature_vector(
        site_encoded_val, pd.Timestamp.today(),
        temperature, humidity, pm10,
        float(defaults["latitude"]), float(defaults["longitude"]),
        float(defaults["pm2_5_lag1"]), float(defaults["pm2_5_roll7"]),
        float(defaults["temp_roll7"]),
    )
    X = artifacts["scaler"].transform(pd.DataFrame([vec], columns=features))
    pm2_5_pred = float(artifacts["rf_reg"].predict(X)[0])
    proba = artifacts["rf_clf"].predict_proba(X)[0]
    pred_class = classes[int(np.argmax(proba))]

    r1, r2 = st.columns(2)
    with r1:
        st.subheader("Predicted PM2.5")
        st.metric("Concentration", f"{pm2_5_pred:.1f} µg/m³")
    with r2:
        st.subheader("Predicted category")
        st.markdown(
            f"<h2 style='color:{CATEGORY_COLORS[pred_class]}'>{pred_class}</h2>",
            unsafe_allow_html=True,
        )

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(category_gauge(pm2_5_pred), width="stretch")
        st.caption("US-EPA breakpoints: Good ≤ 12.0 · Moderate ≤ 35.4 · Unhealthy > 35.4 µg/m³")
    with g2:
        st.plotly_chart(probability_bars(classes, proba), width="stretch")
        st.caption("Class probabilities from the tuned Random Forest classifier.")

    st.info(
        f"Site {site} supplies location and the typical 7-day history "
        f"(last PM2.5: {defaults['pm2_5_lag1']:.0f} µg/m³, 7-day mean: "
        f"{defaults['pm2_5_roll7']:.0f} µg/m³) used by the model."
    )

# ---------------------------------------------------------------------------
# Tab 8 — Key Findings & Conclusion
# ---------------------------------------------------------------------------
with tabs[8]:
    st.markdown("## 8. Key Findings & Conclusion")
    for finding in [
        "**PM2.5 is highly predictable** from heterogeneous sensor data — R² ≈ 0.95, "
        "RMSE ≈ 2 µg/m³ on held-out data.",
        "**Lagged and rolling PM2.5 features dominate** (`pm2_5_lag1`, `pm2_5_roll7`, "
        "`pm10`) — continuous monitoring matters more than weather alone.",
        "**Seasonality is the strongest exogenous driver** — the Jun–Aug peak is "
        "captured by monthly cycling and temperature.",
        "**Space matters little** once features are present — pollution levels are "
        "broadly similar across Nairobi's sites given the same weather and history.",
        "**SMOTE + threshold tuning** improved coverage of the rare `Good` and "
        "`Unhealthy` classes without hurting overall accuracy.",
        "**High-pollution days are under-predicted** — residuals concentrate at the "
        "extremes, so the tool is safest as a general monitoring aid, not a spike alert.",
    ]:
        st.markdown(f"- {finding}")

    st.markdown("---")
    st.markdown(
        "**Conclusion** — the pipeline reliably fills monitoring gaps at under-covered "
        "sites and provides a foundation for forward-looking air quality forecasts. "
        "This app is the deployment layer on top of that work."
    )