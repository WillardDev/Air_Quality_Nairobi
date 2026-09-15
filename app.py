import json as _json
import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

import charts

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

CATEGORY_COLORS = {"Good": "#2e8b57", "Moderate": "#d4a017", "Unhealthy": "#b22222"}

_TOP_N_CELLS = ["eda_site_coverage", "2168cff4"]

_LAYOUT_CSS = """
<style>
/* full-width content container */
[data-testid="stAppViewBlockContainer"] {
    max-width: 1560px;
    margin: 0 auto;
    padding-top: 1.6rem;
    padding-bottom: 3.5rem;
}
.container-full {
    width: 100%;
    max-width: 1480px;
    margin: 0 auto;
}

/* justify the written content */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] blockquote,
[data-testid="stMarkdownContainer"] li {
    text-align: justify;
}

/* charts: centered and at a comfortable smaller size */
[data-testid="stPlotlyChart"] {
    width: 82% !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
[data-testid="stImage"] {
    margin-left: auto !important;
    margin-right: auto !important;
}

/* chart headings follow the chart block, not the app column */
.section-chart-head {
    text-align: center;
    margin-top: 0.4rem;
}
.section-chart-desc {
    text-align: center;
    font-size: 0.9rem;
    color: #64748b;
}
.section-chart-insight {
    max-width: 860px;
    margin: 0.5rem auto 0;
    font-size: 0.9rem;
    color: var(--st-text-color, inherit);
}
/* Streamlit 1.5x sets plotly wrappers to width:82%, overflows its own
   overflow:hidden chart div and clips the right side. Restore 100%. */
body [data-testid="stPlotlyChart"] { width: 100% !important; }

/* chart insight card (below the figure, right-aligned) */
.chart-insight-card {
    border-left: 4px solid #2563eb;
    background: rgba(127, 127, 127, 0.07);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--st-text-color, inherit);
    max-width: 560px;
    margin: 0.6rem 0 0 auto;
}
.chart-insight-card p {
    margin: 0 0 0.6rem;
}
.chart-insight-card p:last-child {
    margin-bottom: 0;
}
.chart-insight-card .ci-label {
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.7;
    margin-bottom: 0.45rem;
}
.chart-insight-card code {
    background: rgba(127, 127, 127, 0.15);
    border-radius: 4px;
    padding: 0.05rem 0.3rem;
    font-size: 0.88em;
}

/* KPI dashboard table */
.kpi-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
    margin: 0.8rem 0 0.4rem;
}
.kpi-table th {
    text-align: left;
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.55rem 0.8rem 0.45rem;
    border-bottom: 2px solid rgba(127, 127, 127, 0.25);
    color: var(--st-text-color, #334155);
}
.kpi-table td {
    padding: 0.55rem 0.8rem 0.45rem;
    border-bottom: 1px solid rgba(127, 127, 127, 0.12);
    vertical-align: top;
    color: var(--st-text-color, #334155);
}
.kpi-table tbody tr:nth-child(even) {
    background: rgba(127, 127, 127, 0.05);
}
.kpi-table .kpi-cat {
    font-weight: 700;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 0.7rem 0.8rem 0.3rem;
    border-bottom: 2px solid rgba(127, 127, 127, 0.18);
    color: #2563eb;
    background: transparent !important;
}
.kpi-table .kpi-val {
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.kpi-table .kpi-note {
    font-size: 0.84rem;
    color: #64748b;
}
.kpi-status {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.kpi-status.green  { background: #16a34a; }
.kpi-status.amber  { background: #d97706; }
.kpi-status.red    { background: #dc2626; }

/* back-to-top floating button (JS toggles .show on scroll) */
#back-to-top {
    position: fixed;
    right: 22px;
    bottom: 22px;
    width: 46px;
    height: 46px;
    border-radius: 50%;
    border: 1px solid transparent;
    background: rgba(15, 23, 42, 0.88);
    color: #ffffff;
    font-size: 20px;
    line-height: 1;
    padding: 0;
    cursor: pointer;
    z-index: 99999;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    opacity: 0;
    transform: translateY(14px) scale(0.92);
    pointer-events: none;
    transition: opacity 0.25s ease, transform 0.25s ease;
    -webkit-appearance: none;
    appearance: none;
}
#back-to-top.show {
    opacity: 1;
    transform: translateY(0) scale(1);
    pointer-events: auto;
}
</style>
"""


_BACK_TO_TOP_JS = r"""
(function () {
  if (window.__aqBackToTop) return;
  window.__aqBackToTop = true;

  var SCROLL_THRESHOLD = 300;
  var doc = document;
  var btn = null;
  var lastTarget = null;

  function scrollTop() {
    if (doc.scrollingElement && doc.scrollingElement.scrollTop > 0)
      return doc.scrollingElement.scrollTop;
    if (window.pageYOffset) return window.pageYOffset;
    var main = doc.querySelector('[data-testid="stMain"]');
    return main ? main.scrollTop : 0;
  }

  function update() {
    if (!btn) return;
    var show = scrollTop() > SCROLL_THRESHOLD;
    btn.classList.toggle('show', show);
  }

  function paint() {
    if (!btn) return;
    var app = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.body;
    var scheme = window.getComputedStyle(app).getPropertyValue('color-scheme').trim();
    var dark = scheme === 'dark' || (!scheme && window.matchMedia &&
      window.matchMedia('(prefers-color-scheme: dark)').matches);
    btn.style.color = dark ? '#16181d' : '#ffffff';
    btn.style.background = dark ? 'rgba(255,255,255,0.92)' : 'rgba(15,23,42,0.88)';
    btn.style.borderColor = dark ? 'rgba(255,255,255,0.4)' : 'rgba(15,23,42,0.25)';
  }

  function init() {
    btn = doc.getElementById('back-to-top');
    if (!btn) return false;
    btn.addEventListener('click', function () {
      var el = null;
      if (lastTarget && lastTarget !== doc && lastTarget.scrollTop !== undefined)
        el = lastTarget;
      if (!el && doc.scrollingElement) el = doc.scrollingElement;
      if (el && el.scrollTo) {
        try { el.scrollTo({ top: 0, behavior: 'smooth' }); }
        catch (e) { el.scrollTop = 0; }
      }
      if (window.scrollTo) {
        try { window.scrollTo({ top: 0, behavior: 'smooth' }); }
        catch (e) {}
      }
    });
    doc.addEventListener('scroll', function (ev) { lastTarget = ev.target; update(); }, true);
    window.addEventListener('resize', update);
    try {
      var app = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.body;
      new MutationObserver(paint).observe(app, { attributes: true });
    } catch (e) {}
    paint();
    update();
    return true;
  }

  if (!init()) {
    var tries = 0;
    var timer = setInterval(function () {
      tries += 1;
      if (init() || tries > 40) clearInterval(timer);
    }, 250);
  }
  window.addEventListener('load', function () { if (btn) update(); });
})();
"""


def _inject_page_js():
    """Run a script in the parent page from a component iframe.

    st.markdown strips <script> tags, but components.html executes script in a
    real iframe; that script can inject a <script> into the parent document.
    """
    payload = (
        "<script>"
        "var p=window.parent.document;"
        "var h=p.head||p.getElementsByTagName('head')[0];"
        "var s=p.createElement('script');"
        f"s.text={_json.dumps(_BACK_TO_TOP_JS)};"
        "h.appendChild(s);"
        "</script>"
    )
    components.html(payload, height=0)


def _insight_html(insight):
    """Turn a two-paragraph insight into the side-card HTML."""
    paras = [p.strip() for p in insight.split("\n\n") if p.strip()]
    if not paras:
        return ""
    body = []
    for p in paras:
        p = re.sub(r"`([^`]+)`", r"<code>\1</code>", p)
        p = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", p)
        body.append(f"<p>{p}</p>")
    return ("<div class='chart-insight-card'><div class='ci-label'>💡 Chart insight</div>"
            + "".join(body) + "</div>")


@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib")),
        "features": joblib.load(os.path.join(MODELS_DIR, "features.joblib")),
        "site_dummy_cols": joblib.load(os.path.join(MODELS_DIR, "site_dummy_cols.joblib")),
        "site_defaults": pd.read_csv(
            os.path.join(MODELS_DIR, "site_defaults.csv")
        ).set_index("site_name"),
        "metrics": _json.load(open(os.path.join(MODELS_DIR, "model_metrics.json"))),
    }


@st.cache_resource
def load_viz_manifest():
    path = os.path.join(PLOTS_DIR, "manifest.json")
    if not os.path.exists(path):
        return []
    return _json.load(open(path))


def build_feature_row(site, datetime_, temperature, humidity, defaults_row,
                      features, site_dummy_cols):
    """Build the 85-feature prediction row, in the exact training-column order.

    Weather comes from the sliders; location and the typical site history come
    from per-site defaults; calendar features are derived from `datetime_`; the
    site identity enters as one 1-of-73 one-hot indicator.
    """
    month = datetime_.month
    day_of_week = datetime_.weekday()
    row = {
        "temperature": temperature,
        "humidity": humidity,
        "latitude": float(defaults_row["latitude"]),
        "longitude": float(defaults_row["longitude"]),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "day_sin": np.sin(2 * np.pi * day_of_week / 7),
        "day_cos": np.cos(2 * np.pi * day_of_week / 7),
        "is_weekend": int(day_of_week >= 5),
        "pm2_5_lag1": float(defaults_row["pm2_5_lag1"]),
        "pm2_5_roll7": float(defaults_row["pm2_5_roll7"]),
        "temp_roll7": float(defaults_row["temp_roll7"]),
    }
    for col in site_dummy_cols:
        row[col] = 1 if col == f"site_{site}" else 0
    return pd.DataFrame([row], columns=features)


def categorize_pm25(val):
    if val <= 12.0:
        return "Good"
    if val <= 35.4:
        return "Moderate"
    return "Unhealthy"


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


def _kpi_table_html(metrics):
    """Build the horizontal KPI table HTML from the persisted model metrics."""
    f = metrics["final"]
    bands = {b["band"].split(" (")[0]: b for b in metrics["error_by_band"]}
    unh = bands["Unhealthy"]
    cells = [
        ("RMSE (held-out)", f"≈ {f['RMSE']:.1f} µg/m³"),
        ("R²", f"≈ {f['R2']:.2f}"),
        ("MAE · Unhealthy band", f"≈ {unh['mean_abs_error']:.0f} µg/m³"),
    ]
    heads = "".join(f"<th>{name}</th>" for name, _ in cells)
    vals = "".join(f"<td class='kpi-val'>{val}</td>" for _, val in cells)
    return (f"<table class='kpi-table'><thead><tr>{heads}"
            f"<th>Positioning</th></tr></thead><tbody><tr>{vals}"
            f"<td class='kpi-note'>A useful monitoring estimate, not a spike-alert "
            f"guarantee — marketed for gap-filling and early advisories, not as a "
            f"single-day spike alarm.</td></tr></tbody></table>")


def render_section_viz(section_number):
    """Render figures with their insight directly below, in notebook order."""
    manifest = load_viz_manifest()
    if not manifest:
        st.warning("No `plots/manifest.json` found — visualizations are unavailable.")
        return
    raw = manifest if isinstance(manifest, dict) else {}
    images = raw.get("images", []) or []
    texts = raw.get("texts", []) or []
    merged = []
    for v in images:
        if v.get("section_num") == section_number:
            merged.append({"kind": "figure", "file": v["file"], "title": v["title"],
                           "insight": v.get("insight", ""), "order": v.get("order", 0)})
    for v in texts:
        if v.get("section_num") == section_number:
            merged.append({"kind": "text", "file": None, "title": v["title"],
                           "insight": v.get("insight", ""), "order": v.get("order", 0)})
    merged.sort(key=lambda e: (tuple(e["order"]) if isinstance(e["order"], list) else (e["order"],),
                               e["kind"] != "figure" and 1 or 0))
    if not merged:
        return
    last_title = None
    for e in merged:
        if e["title"] != last_title:
            last_title = e["title"]
        if e["kind"] == "figure":
            cell_id = e["file"].removesuffix(".png")
            st.markdown(f"<h5 class='section-chart-head'>{e['title']}</h5>",
                        unsafe_allow_html=True)
            if cell_id in _TOP_N_CELLS:
                n = st.slider("Top N sites shown",
                              min_value=3, max_value=74, value=10, step=1,
                              key=f"topn_{cell_id}")
            else:
                n = 10
            fig = charts.get_figure(cell_id, top_n=n)
            if fig is not None:
                st.plotly_chart(fig, width=1180)
            else:
                st.image(os.path.join(PLOTS_DIR, e["file"]))
            if e["insight"]:
                st.markdown(_insight_html(e["insight"]), unsafe_allow_html=True)
        elif e["insight"]:
            st.markdown(f"📌 {e['insight']}")
        st.divider()


st.set_page_config(page_title="Nairobi Air Quality — PM2.5 Prediction", layout="wide")
st.markdown(_LAYOUT_CSS, unsafe_allow_html=True)
st.markdown('<button id="back-to-top" type="button" aria-label="Back to top">↑</button>',
            unsafe_allow_html=True)
_inject_page_js()

artifacts = load_artifacts()
model = artifacts["model"]
features = artifacts["features"]
site_dummy_cols = artifacts["site_dummy_cols"]
site_defaults = artifacts["site_defaults"]
metrics = artifacts["metrics"]

tabs = st.tabs([
    "1. Problem Statement",
    "2. Business Use Case & Impact",
    "3. Data Understanding",
    "4. Exploratory Data Analysis",
    "5. Feature Engineering and Selection",
    "6. Model Selection & Modelling",
    "7. Model Evaluation",
    "8. Error Analysis",
    "9. Model Explainability",
    "10. Hyperparameter Tuning",
    "11. Final Model & Recommendations",
    "12. Project Demonstration",
])

# ---------------------------------------------------------------------------
# Tab 1 — Problem Statement
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.title("Nairobi Air Quality — PM2.5 Prediction")
        st.caption("A full machine learning pipeline on AirQo sensor data (2025)")

    st.markdown("## 1. Problem Statement")
    st.markdown(
        "Nairobi runs one of the densest low-cost air-quality monitoring networks in "
        "Africa (74 AirQo sites, continuous 2025 data), yet a raw sensor feed leaves "
        "air-quality management three problems unsolved:"
    )
    for item in [
        "**Coverage gaps** — missing sensor readings (573–633 cells per column) and "
        "sparsely reporting sites leave the daily pollution picture incomplete.",
        "**Passive data** — raw readings don't say which locations are worst, what "
        "drives pollution, when it peaks, or what levels to expect next days.",
        "**No exposure translation** — a µg/m³ number alone doesn't tell a resident "
        "or an agency the health band (Good / Moderate / Unhealthy).",
    ]:
        st.markdown(f"- {item}")

    st.markdown(
        "**Task** — given a site's spatio-temporal context (location, date, weather, "
        "and its own recent sensor history), estimate daily PM2.5 (µg/m³) as a "
        "**regression** (continuous value), scored with RMSE / MAE / R². The "
        "Good/Moderate/Unhealthy label is a fixed US-EPA threshold applied *after* "
        "the regression, not a separately trained classifier. (Logistic Regression, "
        "a classifier, is explicitly out of scope for this project.)"
    )

    left, right = st.columns(2)
    with left:
        st.markdown("### Project objectives")
        for o in [
            "Analyze the distribution and variation of PM2.5 across Nairobi's sites.",
            "Explore how temperature, humidity, and season relate to air quality.",
            "Build predictive regression models for daily PM2.5.",
            "Evaluate performance honestly and identify the most important features.",
            "Provide actionable insight for monitoring and public health.",
        ]:
            st.markdown(f"- {o}")
    with right:
        st.markdown("### Research questions")
        for q in [
            "What spatial and temporal patterns shape PM2.5 across Nairobi?",
            "How much do weather and season explain — and how much does recent history?",
            "Can we predict daily PM2.5 well enough to be useful for monitoring?",
            "Which sites are most polluted and why?",
            "How does PM2.5 vary across seasons in Nairobi?",
        ]:
            st.markdown(f"- {q}")

    st.markdown("### Pipeline at a glance")
    for step, desc in [
        ("🧹 Data Understanding", "cleaning, zero-exposure, imputation (§3)"),
        ("🔍 EDA", "distributions, seasonality, site variation (§4)"),
        ("🧊 Feature Engineering and Selection", "encoding, feature engineering, selection (§5)"),
        ("⚖️ Model Selection & Modelling", "five regression models (§6)"),
        ("📊 Model Evaluation", "RMSE/MAE/R² on held-out data (§7)"),
        ("🩺 Error Analysis", "residuals, error by PM2.5 range (§8)"),
        ("💡 Explainability", "importance, permutation, partial dependence (§9)"),
        ("🚀 Project Demonstration", "interactive Streamlit predictor (§12)"),
    ]:
        st.markdown(f"**{step}** — {desc}")

# ---------------------------------------------------------------------------
# Tab 2 — Business Use Case & Impact
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("## 2. Business Use Case & Impact")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Who needs this")
        for item in [
            "**AirQo & city agencies** — fill missing site-days and flag faulty "
            "sensors by comparing readings against predictions.",
            "**Health & environment authorities** — same-day PM2.5 estimates plus a "
            "Good / Moderate / Unhealthy health band for public advisories.",
            "**Residents** — the interactive app (§12) turns the model into instant "
            "predictions for any site.",
        ]:
            st.markdown(f"- {item}")
    with c2:
        st.subheader("Impact")
        for item in [
            "**Cost** — fills monitoring gaps with existing data, no new hardware needed.",
            "**Timeliness** — moves from retrospective reporting to same-day estimates.",
            "**Public health** — a transparent, honest estimate of what to expect, with "
            "its limits stated clearly (§8, §11).",
        ]:
            st.markdown(f"- {item}")
    st.markdown("### KPIs")
    st.markdown(_kpi_table_html(metrics), unsafe_allow_html=True)
    st.info(
        "**Deployment path** — the winning XGBoost model is exported to `models/` and "
        "served by this Streamlit app (see tab 12)."
    )

# ---------------------------------------------------------------------------
# Tab 3 — Data Understanding
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("## 3. Data Understanding")
    st.markdown(
        "One year (2025) of daily readings: **16,374 rows × 11 columns** across "
        "**74 sites** — site name, datetime, coordinates, temperature, humidity, "
        "and PM2.5. PM10 was dropped from the analysis (see cleaning notes)."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sites", "74")
    c2.metric("Rows (raw)", "16,374")
    c3.metric("Coverage", "Full year 2025")
    c4.metric("Rows (modeled)", "15,778")

    st.markdown("### What was cleaned")
    for item in [
        "Dropped constant columns (`frequency`, `network`, `site_id`, `device_name`)",
        "Zeroed temperature/humidity (≈6.3% of rows — a broken shared probe) and "
        "zeroed PM2.5 readings treated as **missing**, not valid calm days",
        "Dropped 596 rows (3.6%) with an unusable `pm2_5` target",
        "Imputed temperature/humidity with a **per-site median**, falling back to the "
        "city-wide median for two sites whose weather probe never worked",
        "Outliers were *flagged* (IQR) but **kept** — they are real pollution "
        "episodes, and removing them would bias the model against spikes",
    ]:
        st.markdown(f"- {item}")

    render_section_viz(3)

# ---------------------------------------------------------------------------
# Tab 4 — Exploratory Data Analysis
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("## 4. Exploratory Data Analysis")
    st.markdown(
        "Distribution shapes, site-by-site variation, seasonal patterns, and the "
        "relationships between weather, history, and PM2.5."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("- Typical reading ≈ **22.7 µg/m³** mean (20.3 median, Moderate range)")
        st.markdown("- Clear **Jun–Aug dry-season peak** (monthly boxplot + trend)")
        st.markdown("- **Recent history is the strongest driver**: 7-day rolling PM2.5 "
                    "correlates r ≈ 0.72 with today's value")
    with right:
        st.markdown("- Coverage is uneven — the best sites log ~360 days, the sparsest "
                    "fewer than 60")
        st.markdown("- Weekday/weekend effect exists but is small (r ≈ −0.04)")
        st.markdown("- Weather is weakly related: temperature r ≈ −0.09, humidity r ≈ 0.01")

    render_section_viz(4)

# ---------------------------------------------------------------------------
# Tab 5 — Feature Engineering and Selection
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("## 5. Feature Engineering and Selection")
    st.markdown(
        "Feature engineering produces **85 features** across four groups: weather & "
        "location, calendar, history, and site identity."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.subheader("Weather & location")
        for f in ["temperature", "humidity", "latitude", "longitude"]:
            st.markdown(f"- `{f}`")
    with c2:
        st.subheader("Calendar")
        for f in ["month_sin", "month_cos", "day_sin", "day_cos", "is_weekend"]:
            st.markdown(f"- `{f}`")
    with c3:
        st.subheader("History (lagged/rolling)")
        for f in ["pm2_5_lag1", "pm2_5_roll7", "temp_roll7"]:
            st.markdown(f"- `{f}`")
    with c4:
        st.subheader("Site identity")
        st.markdown("73 one-hot `site_*` columns (74 sites, `drop_first=True` reference)")

    st.markdown(
        "Cyclical sin/cos encoding keeps month and day-of-week distances meaningful; "
        "1-day lag and 7-day rolling PM2.5 plus rolling temperature capture each "
        "site's own recent history (computed per site, sorted by date, so nothing "
        "leaks across sites or into the future). Sites are **one-hot** encoded — "
        "arbitrary numeric site codes would imply a false ordering that linear "
        "models would exploit."
    )

    render_section_viz(5)

    st.markdown("### What drives predictions?")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Feature importance** (XGBoost):\n"
            "- `pm2_5_roll7` (7-day mean)\n"
            "- `pm2_5_lag1` (yesterday)\n"
            "- `month_cos` (seasonality)\n\n"
            "Recent readings carry most of the signal."
        )
    with right:
        st.markdown(
            "**History beats weather** — shuffling `pm2_5_lag1` costs R² ≈ 0.55, more "
            "than every other feature combined (§9).\n\n"
            "**Seasonality** is the strongest exogenous driver.\n\n"
            "**Space enters through the site columns** — individual coordinates are "
            "weak, but the 73 site indicators jointly encode real location differences."
        )

# ---------------------------------------------------------------------------
# Tab 6 — Model Selection & Modelling
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("## 6. Model Selection & Modelling")
    st.markdown(
        "Five regression models run on the same **85 features** with one **80/20 "
        "random split (seed 42)** for a fair comparison. The scope asks for "
        "regression only, so Logistic Regression — a *classifier* — is out. Linear "
        "baselines receive standardized inputs (§6.2); the tree models are "
        "scale-invariant and train on the raw features."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Models compared")
        st.markdown(
            "- Linear Regression\n"
            "- Lasso (α = 0.1)\n"
            "- Ridge (α = 1.0)\n"
            "- Random Forest Regressor (150 trees, depth 20)\n"
            "- **XGBoost Regressor** (200 trees, depth 6, lr 0.1)"
        )
    with c2:
        st.subheader("Winner by RMSE")
        st.markdown(
            f"**XGBoost** — RMSE {metrics['comparison']['XGBoost']['RMSE']:.2f}, "
            f"MAE {metrics['comparison']['XGBoost']['MAE']:.2f}, "
            f"R² {metrics['comparison']['XGBoost']['R2']:.2f}.\n\n"
            f"Random Forest is close behind (RMSE "
            f"{metrics['comparison']['Random Forest']['RMSE']:.2f}); the three linear "
            "baselines cluster around RMSE ≈ 6.7–6.8."
        )

    st.markdown("### Why XGBoost?")
    for item in [
        "Gradient boosting captures the non-linear, seasonal, and site-specific "
        "structure of the data",
        "Handles the 73-column one-hot design naturally, and is scale-invariant",
        "Robust to the sparse-coverage patterns of low-cost sensor networks",
        "Marginal, honest edge over Random Forest that §7 confirms on held-out data",
    ]:
        st.markdown(f"- {item}")

    render_section_viz(6)

# ---------------------------------------------------------------------------
# Tab 7 — Model Evaluation
# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown("## 7. Model Evaluation")
    st.markdown(
        "Every model is scored on the same held-out **20% of site-days (3,156 rows)** "
        "that no model saw during training or tuning. Regression: RMSE / MAE / R²."
    )

    eval_rows = [
        {
            "Model": name,
            "RMSE (µg/m³)": f"{m['RMSE']:.3f}",
            "MAE (µg/m³)": f"{m['MAE']:.3f}",
            "R²": f"{m['R2']:.3f}",
        }
        for name, m in sorted(
            metrics["comparison"].items(), key=lambda kv: kv[1]["RMSE"]
        )
    ]
    st.dataframe(pd.DataFrame(eval_rows), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Winner")
        st.markdown(
            "**XGBoost Regressor** (untuned §6 defaults)\n"
            f"- RMSE {metrics['comparison']['XGBoost']['RMSE']:.2f} µg/m³\n"
            f"- MAE {metrics['comparison']['XGBoost']['MAE']:.2f} µg/m³\n"
            f"- R² {metrics['comparison']['XGBoost']['R2']:.2f}"
        )
    with c2:
        st.subheader("Honest interpretation")
        st.markdown(
            "Relative to the *actual* range of the data these are real errors — a "
            "typical day (~20 µg/m³) is predicted within ~±4 µg/m³. The R² ≈ 0.66 "
            "means the model explains about two-thirds of the variance: useful for "
            "monitoring, not for claiming near-perfect prediction."
        )

    render_section_viz(7)

# ---------------------------------------------------------------------------
# Tab 8 — Error Analysis
# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("## 8. Error Analysis")
    st.markdown(
        "A closer look at *where* the winning XGBoost model is wrong: residuals and "
        "error segmented by PM2.5 band."
    )

    band_rows = []
    for b in metrics["error_by_band"]:
        band_rows.append({
            "PM2.5 band": b["band"],
            "Site-days": int(b["count"]),
            "Mean error (µg/m³)": f"{b['mean_error']:.2f}",
            "Mean |error| (µg/m³)": f"{b['mean_abs_error']:.2f}",
        })
    st.dataframe(pd.DataFrame(band_rows), use_container_width=True)
    st.markdown(
        "- **Good (≤12) and Moderate days** are predicted within ~3–4 µg/m³ on "
        "average — the model is well calibrated where most data lives.\n"
        "- **Unhealthy (>35.4) days are systematically under-predicted** — mean "
        "*signed* error ≈ −8.6 µg/m³, MAE ≈ 10.3. The model is trained mostly on "
        "typical days and is conservative about real spikes.\n"
        "- This is a **known, stated limitation**, not a reason to distrust the whole "
        "tool: it still beats 'no prediction at all' for the sparsest sites."
    )
    render_section_viz(8)

# ---------------------------------------------------------------------------
# Tab 9 — Model Explainability
# ---------------------------------------------------------------------------
with tabs[8]:
    st.markdown("## 9. Model Explainability")
    st.markdown(
        "Two lenses on why the model makes its predictions: built-in feature "
        "importances (split usage) and **permutation importance** (the actual "
        "predictive value — how much held-out R² drops when each feature is "
        "shuffled), plus partial-dependence curves for the key features."
    )
    st.markdown(
        "**Caveat:** importance measures *association*, not causation. A high-ranking "
        "site indicator means location is a strong predictor — it stands in for "
        "unmeasured local factors like traffic density."
    )
    render_section_viz(9)

# ---------------------------------------------------------------------------
# Tab 10 — Hyperparameter Tuning
# ---------------------------------------------------------------------------
with tabs[9]:
    st.markdown("## 10. Hyperparameter Tuning")
    st.markdown(
        "The winner is tuned with **GridSearchCV (5-fold)** on the training set "
        "only; the held-out test set is touched exactly once, afterwards, to make "
        "the honest before/after call."
    )
    tuning_rows = [
        {"Model": "XGBoost Regressor",
         "Search space": "`n_estimators` ∈ {200, 300}, `max_depth` ∈ {4, 6, 8}, "
                         "`learning_rate` ∈ {0.05, 0.1}",
         "Best parameters": "`learning_rate=0.05, max_depth=6, n_estimators=200`",
         "Best CV RMSE (train folds)": f"{metrics['tuned']['cv_rmse']:.2f}",
         "Held-out RMSE before/after": (
             f"{metrics['untuned']['RMSE']:.3f} → {metrics['tuned']['RMSE']:.3f}"),
         },
    ]
    st.dataframe(pd.DataFrame(tuning_rows), use_container_width=True)

    st.subheader("Held-out test — before vs after tuning")
    before_after = pd.DataFrame(
        {
            "RMSE (µg/m³)": [metrics["untuned"]["RMSE"], metrics["tuned"]["RMSE"]],
            "MAE (µg/m³)": [metrics["untuned"]["MAE"], metrics["tuned"]["MAE"]],
            "R²": [metrics["untuned"]["R2"], metrics["tuned"]["R2"]],
        },
        index=["Before tuning", "After tuning"],
    )
    st.dataframe(before_after.round(4), use_container_width=True)
    st.markdown(
        "Tuning is **flat** — held-out RMSE actually edges *up* from "
        f"{metrics['untuned']['RMSE']:.3f} to {metrics['tuned']['RMSE']:.3f} "
        "(≈ +0.07%), with MAE and R² barely moving. All three deltas sit well "
        "inside the metric's own noise, so tuning gains nothing measurable. That is "
        "expected: the search grid is deliberately narrow and the §6 defaults "
        "(200 trees, depth 6, lr 0.1) already sit at its optimum, leaving no "
        "headroom for the hyperparameters to exploit. Per §10's decision rule the "
        "gain is negligible, so the simpler **untuned §6 model is kept as final** — "
        "tuning would only add complexity without improving predictions."
    )

# ---------------------------------------------------------------------------
# Tab 11 — Final Model & Recommendations
# ---------------------------------------------------------------------------
with tabs[10]:
    st.markdown("## 11. Final Model & Recommendations")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Champion model")
        st.markdown(
            "**XGBoost Regressor** (untuned §6 defaults)\n\n"
            f"- RMSE ≈ {metrics['final']['RMSE']:.1f} µg/m³\n"
            f"- MAE ≈ {metrics['final']['MAE']:.1f} µg/m³\n"
            f"- R² ≈ {metrics['final']['R2']:.2f}\n\n"
            "Trained on 12,622 site-days × 85 features, exported to `models/`, and "
            "served by this app."
        )
    with c2:
        st.subheader("Recommendations")
        for item in [
            "Deploy the regressor to fill coverage gaps at the sparsely reporting "
            "sites (§4.2) and provide same-day estimates ahead of official readings.",
            "Treat the model as a **monitoring aid, not a spike alarm** — high days "
            "are its weakest point (§8).",
            "Protect sensor continuity: `pm2_5_lag1` is the single most valuable "
            "feature (permutation importance §9).",
            "For the future: add traffic, wind, and land-use data; retrain quarterly; "
            "use a **chronological** split to test true forecasting.",
        ]:
            st.markdown(f"- {item}")
    st.markdown("### Key findings")
    for finding in [
        "**PM2.5 is meaningfully predictable** from weather, location, and history — "
        "RMSE ≈ 6.3 µg/m³, R² ≈ 0.66 on held-out data.",
        "**Recent history dominates** (`pm2_5_roll7`, `pm2_5_lag1`) — what happened "
        "at a site in the last week predicts today better than any weather variable.",
        "**Seasonality is the strongest exogenous driver** — the Jun–Aug peak is "
        "captured through cyclical month encoding.",
        "**Location enters through site indicators** — coordinates alone are weak, "
        "but 73 one-hot site columns encode the real spatial structure (§4.7).",
        "**High-pollution days are under-predicted** — error concentrates at the "
        "extremes, so the tool is clearest as a general monitoring and gap-filling "
        "aid.",
        "**No traffic/wind/land-use data yet** — adding these is the most promising "
        "path to fixing the spike under-prediction.",
    ]:
        st.markdown(f"- {finding}")

    st.markdown("---")
    st.markdown(
        "**Conclusion** — the pipeline delivers an end-to-end, reproducible flow from "
        "cleaning raw AirQo data (including the honest treatment of fake-zero "
        "readings and retained outliers) through feature engineering, model "
        "selection, error analysis, and explainability. XGBoost is the recommended "
        "winner: it captures the non-linear, seasonal, history-driven structure of "
        "Nairobi's PM2.5 and is robust to the sparse-coverage problems inherent in "
        "low-cost sensor networks. The result fills monitoring gaps at under-covered "
        "sites and provides honest same-day estimates — with its limitations stated "
        "plainly."
    )

# ---------------------------------------------------------------------------
# Tab 12 — Project Demonstration: Interactive Predictor
# ---------------------------------------------------------------------------
with tabs[11]:
    st.markdown("## 12. Project Demonstration — Interactive Predictor")
    st.caption(
        "Pick a site and current weather; location and the site's typical 7-day "
        "history come from per-site defaults. The exported XGBoost model predicts "
        "today's PM2.5."
    )

    col_site, col_temp, col_hum = st.columns(3)
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

    X = build_feature_row(site, pd.Timestamp.today(), temperature, humidity,
                          defaults, features, site_dummy_cols)
    pm2_5_pred = float(model.predict(X)[0])
    pred_class = categorize_pm25(pm2_5_pred)

    r1, r2 = st.columns(2)
    with r1:
        st.subheader("Predicted PM2.5")
        st.metric("Concentration", f"{pm2_5_pred:.1f} µg/m³")
    with r2:
        st.subheader("Predicted health band")
        st.markdown(
            f"<h2 style='color:{CATEGORY_COLORS[pred_class]}'>{pred_class}</h2>",
            unsafe_allow_html=True,
        )

    st.plotly_chart(category_gauge(pm2_5_pred), width=1180)
    st.caption(
        "US-EPA breakpoints: Good ≤ 12.0 · Moderate ≤ 35.4 · Unhealthy > 35.4 µg/m³. "
        "The band is derived from the predicted value, not a separately trained "
        "classifier."
    )

    st.info(
        f"Site **{site}** supplies location and the typical 7-day history "
        f"(yesterday's PM2.5: {defaults['pm2_5_lag1']:.0f} µg/m³, 7-day mean: "
        f"{defaults['pm2_5_roll7']:.0f} µg/m³) used by the model. For the two sites "
        "whose weather probe never recorded valid data, temperature/humidity "
        "defaults are the city-wide median (least trustworthy here). "
        "The exported model, feature list, and per-site defaults live in `models/`."
    )