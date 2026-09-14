"""Interactive Plotly charts for the Streamlit app.

Every chart from the regression pipeline has an equivalent interactive figure
here, keyed by the same cell id used in ``plots/manifest.json``. Chart data is
read from ``plots/chart_data.json`` (precomputed by ``rebuild_artifacts.py``),
so no models or sklearn imports are needed at runtime. Fall back to the PNG via
``get_figure`` returning ``None``.
"""

import json
import os
from functools import lru_cache

import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
_CACHE_PATH = os.path.join(_PLOTS_DIR, "chart_data.json")

_PALETTE = [
    "#2563eb", "#f59e0b", "#10b981", "#8b5cf6",
    "#ef4444", "#0ea5e9", "#f97316", "#64748b",
]
_NUMERIC_LABELS = {
    "latitude": "Latitude", "longitude": "Longitude",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
    "pm2_5": "PM2.5 (µg/m³)",
}
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@lru_cache(maxsize=1)
def _data():
    if not os.path.exists(_CACHE_PATH):
        return None
    with open(_CACHE_PATH) as f:
        return json.load(f)


def _style(fig, height=560, showlegend=None, xtitle=None, ytitle=None):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=30, t=30, b=55),
        hoverlabel=dict(namelength=-1, font=dict(size=12)),
    )
    fig.update_yaxes(showgrid=False, title=None)
    if xtitle:
        fig.update_xaxes(title_text=xtitle)
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    if showlegend is None:
        showlegend = len(fig.data) > 1
    fig.update_layout(showlegend=showlegend)
    return fig


def _box_trace(values, label, color):
    hover = (f"<b>{label}</b><br>median %{{median:.2f}} · mean %{{mean:.2f}}"
             f"<br>Q1 %{{q1:.2f}} · Q3 %{{q3:.2f}}"
             f"<br>min %{{min:.2f}} · max %{{max:.2f}}<br>n = %{{count}}<extra></extra>")
    return go.Box(y=values, name=label, marker_color=color, boxpoints=False,
                  hoveron="boxes", hovertemplate=hover)


def _fig_eda_site_coverage(d, top_n=None):
    top_n = d.get("_top_n", 10) if top_n is None else top_n
    pairs = sorted(zip(d["site_counts"]["site"], d["site_counts"]["count"]),
                   key=lambda t: t[1], reverse=True)[:top_n]
    sites = [s for s, _ in pairs]
    counts = [c for _, c in pairs]
    fig = go.Figure(go.Bar(y=sites, x=counts, orientation="h",
                           marker_color="#2563eb",
                           hovertemplate="<b>%{y}</b><br>readings: %{x:,}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=sites)
    return _style(fig, height=max(400, top_n * 40 + 100), xtitle="Number of readings")


def _fig_39877be1(d):
    cols = d["corr"]["cols"]
    z = d["corr"]["values"]
    text = [[f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(x=cols, y=cols, z=z, text=text, texttemplate="%{text}",
                               colorscale="RdBu", zmin=-1, zmax=1, showscale=False,
                               hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>"))
    fig.update_xaxes(tickangle=-45, nticks=len(cols))
    fig.update_yaxes(nticks=len(cols), autorange="reversed")
    return _style(fig, height=540)


def _fig_3d96bbad(d):
    e = d["eda"]
    cols = ["temperature", "humidity", "pm2_5"]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[_NUMERIC_LABELS[c] for c in cols],
                        horizontal_spacing=0.07)
    for i, col in enumerate(cols):
        fig.add_trace(go.Histogram(x=e[col], nbinsx=60,
                                   marker_color=_PALETTE[i], showlegend=False,
                                   hovertemplate=f"<b>{_NUMERIC_LABELS[col]}</b>"
                                                 "<br>%{x:.2f} → %{y} readings<extra></extra>"),
                      1, i + 1)
        fig.update_xaxes(row=1, col=i + 1, title=None)
        fig.update_yaxes(row=1, col=i + 1, title=None)
    return _style(fig, height=440)


def _fig_weekend_plot(d):
    e = d["eda"]
    fig = go.Figure()
    for lbl, mask, col in [("Weekday", 0, "#2563eb"), ("Weekend", 1, "#f59e0b")]:
        vals = [v for v, w in zip(e["pm2_5"], e["is_weekend"]) if int(w) == mask]
        fig.add_trace(_box_trace(vals, lbl, col))
    return _style(fig, height=480, ytitle="PM2.5 (µg/m³)")


def _fig_2168cff4(d, top_n=None):
    top_n = d.get("_top_n", 10) if top_n is None else top_n
    s = d["site_pm25"]
    pairs = sorted(zip(s["site"], s["pm25"]), key=lambda t: t[1], reverse=True)[:top_n]
    sites = [x for x, _ in pairs]
    pm25 = [v for _, v in pairs]
    fig = go.Figure(go.Bar(y=sites, x=pm25, orientation="h",
                           marker_color="#7a1f1f",
                           hovertemplate="<b>%{y}</b><br>avg PM2.5 %{x:.2f} µg/m³<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=sites)
    return _style(fig, height=max(400, top_n * 40 + 100), xtitle="Mean PM2.5 (µg/m³)")


def _fig_a2be3f89(d):
    e = d["eda"]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=e["temperature"], y=e["pm2_5"], mode="markers",
                               marker=dict(size=4, color="#d97706", opacity=0.45),
                               customdata=list(zip(e["site_name"], e["humidity"])),
                               hovertemplate="PM2.5 %{y:.1f} · %{x:.1f} °C<br>%{customdata[0]}"
                                             "<br>hum %{customdata[1]:.0f}%<extra></extra>"))
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False,
                       text="r ≈ −0.09", font=dict(size=13, color="#475569"))
    return _style(fig, height=520, xtitle="Temperature (°C)", ytitle="PM2.5 (µg/m³)")


def _fig_90ec9e9f(d):
    e = d["eda"]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=e["humidity"], y=e["pm2_5"], mode="markers",
                               marker=dict(size=4, color="#0e7490", opacity=0.45),
                               customdata=list(zip(e["site_name"], e["temperature"])),
                               hovertemplate="PM2.5 %{y:.1f} · %{x:.0f}% RH<br>%{customdata[0]}"
                                             "<br>temp %{customdata[1]:.1f} °C<extra></extra>"))
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False,
                       text="r ≈ 0.01", font=dict(size=13, color="#475569"))
    return _style(fig, height=520, xtitle="Humidity (%)", ytitle="PM2.5 (µg/m³)")


def _fig_2873277e(d):
    e = d["eda"]
    fig = go.Figure()
    for m in range(1, 13):
        vals = [v for v, mm in zip(e["pm2_5"], e["month"]) if int(mm) == m]
        fig.add_trace(_box_trace(vals, _MONTH_NAMES[m - 1], _PALETTE[(m - 1) % len(_PALETTE)]))
    fig.update_xaxes(tickangle=0)
    return _style(fig, height=520, ytitle="PM2.5 (µg/m³)")


def _fig_corr_target(d):
    ct = d["corr_target"]
    feats = ct["features"][::-1]
    vals = ct["values"][::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=feats, x=vals, orientation="h",
                         marker_color=["#10b981" if v >= 0 else "#b23b3b" for v in vals],
                         hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#94a3b8", width=1.5, dash="dot"))
    fig.update_yaxes(categoryorder="array", categoryarray=feats)
    return _style(fig, height=560, xtitle="Correlation with PM2.5")


def _fig_reg_viz(d):
    rr = d["reg_results"]
    metrics_row = [("RMSE", rr["RMSE"], "(µg/m³, lower is better)"),
                   ("MAE", rr["MAE"], "(µg/m³, lower is better)"),
                   ("R²", rr["R2"], "(higher is better)")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[m[0] for m in metrics_row],
                        horizontal_spacing=0.07)
    for k, (name, vals, _) in enumerate(metrics_row):
        fig.add_trace(
            go.Bar(x=rr["models"], y=vals, name=name,
                   marker_color=[("#b45309" if m == "XGBoost" else _PALETTE[i % len(_PALETTE)])
                                 for i, m in enumerate(rr["models"])],
                   text=[f"{v:.3f}" if name in ("RMSE", "MAE") else f"{v:.3f}" for v in vals],
                   textposition="outside",
                   showlegend=False,
                   hovertemplate=f"<b>%{{x}}</b><br>{name} %{{y:.4f}}<extra></extra>"),
            1, k + 1)
        fig.update_xaxes(row=1, col=k + 1, tickangle=-15)
    return _style(fig, height=460)


def _fig_reg_diag(d):
    rd = d["reg_diag"]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Actual vs predicted", "Residuals by prediction"],
                        horizontal_spacing=0.10)
    fig.add_trace(go.Scattergl(x=rd["actual"], y=rd["pred"], mode="markers",
                               marker=dict(size=4, color="#2563eb", opacity=0.5),
                               hovertemplate="actual %{x:.2f} · predicted %{y:.2f}<extra></extra>"),
                  1, 1)
    lim = [min(min(rd["actual"]), min(rd["pred"])), max(max(rd["actual"]), max(rd["pred"]))]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="identity",
                             line=dict(color="#dc2626", width=2, dash="dot"),
                             hovertemplate="Perfect prediction<extra></extra>"), 1, 1)
    fig.add_trace(go.Scattergl(x=rd["pred"], y=rd["resid"], mode="markers",
                               marker=dict(size=4, color="#10b981", opacity=0.5),
                               hovertemplate="predicted %{x:.2f}<br>residual %{y:.2f}<extra></extra>"),
                  1, 2)
    fig.add_hline(y=0, line=dict(color="#dc2626", width=1.5, dash="dot"), row=1, col=2)
    fig.update_xaxes(row=1, col=1, title_text="Actual PM2.5")
    fig.update_yaxes(row=1, col=1, title_text="Predicted PM2.5")
    fig.update_xaxes(row=1, col=2, title_text="Predicted PM2.5")
    fig.update_yaxes(row=1, col=2, title_text="Residual (actual − predicted)")
    return _style(fig, height=500, showlegend=False)


def _fig_feat_imp(d):
    fi = d["feat_imp"]
    feats = fi["features"][::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=feats, x=fi["values"][::-1], orientation="h",
                         marker_color="#2563eb",
                         hovertemplate="<b>%{y}</b><br>importance %{x:.4f}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=feats)
    return _style(fig, height=620, xtitle="XGBoost built-in importance")


def _fig_perm_imp(d):
    pi = d["perm_imp"]
    feats = pi["features"][::-1]
    means = pi["mean"][::-1]
    stds = pi["std"][::-1]
    fig = go.Figure(go.Bar(y=feats, x=means, orientation="h", marker_color="#7c3aed",
                           error_x=dict(type="data", array=stds, thickness=1.2, color="#4c1d95"),
                           customdata=[[s] for s in stds],
                           hovertemplate="<b>%{y}</b><br>R² drop %{x:.3f} ± %{customdata[0]:.3f}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=feats)
    return _style(fig, height=620, xtitle="Held-out R² drop when shuffled")


def _fig_pdp_reg(d):
    pdp = d["pdp_reg"]["series"]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[s["feature"] for s in pdp],
                        horizontal_spacing=0.10)
    for k, s in enumerate(pdp):
        fig.add_trace(go.Scatter(x=s["values"], y=s["average"], mode="lines+markers",
                                 name=s["feature"], line=dict(color=_PALETTE[k], width=2.5),
                                 hovertemplate="<b>%{fullData.name}</b> = %{x:.2f}<br>predicted PM2.5 %{y:.2f}<extra></extra>"),
                      1, k + 1)
        fig.update_xaxes(row=1, col=k + 1, title_text=s["feature"])
        fig.update_yaxes(row=1, col=k + 1, title_text="Average predicted PM2.5")
    return _style(fig, height=460, showlegend=False)


_HANDLERS = {
    "eda_site_coverage": _fig_eda_site_coverage,
    "39877be1": _fig_39877be1,
    "3d96bbad": _fig_3d96bbad,
    "weekend_plot": _fig_weekend_plot,
    "2168cff4": _fig_2168cff4,
    "a2be3f89": _fig_a2be3f89,
    "90ec9e9f": _fig_90ec9e9f,
    "2873277e": _fig_2873277e,
    "corr_target": _fig_corr_target,
    "reg_viz": _fig_reg_viz,
    "reg_diag": _fig_reg_diag,
    "feat_imp": _fig_feat_imp,
    "perm_imp": _fig_perm_imp,
    "pdp_reg": _fig_pdp_reg,
}


def get_figure(cell_id, top_n=10):
    """Return a plotly figure for ``cell_id`` (the PNG stem), or None."""
    handler = _HANDLERS.get(cell_id)
    if handler is None:
        return None
    data = _data()
    if data is None:
        return None
    data["_top_n"] = top_n
    try:
        return handler(data)
    except Exception:
        return None