"""
pages/2_Pressure_Analysis.py – SUPRR Selector | Pressure Detection Engine
Interactive exploration of BPI and BoPI across formats, phases, and custom weights.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pressure Analysis | SUPRR Selector",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0A0F1E; }
    .stApp * { color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #0D1527; }
    .metric-card {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00D4A1; }
    .metric-label { font-size: 0.85rem; color: #A0ADB8; margin-top: 4px; }
    .rank-badge {
        background: linear-gradient(135deg, #00D4A1, #4A9EFF);
        border-radius: 50%;
        width: 32px; height: 32px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 32px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1E2F4A;
    }
    .info-box {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-left: 4px solid #4A9EFF;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.88rem;
        color: #A0ADB8;
        line-height: 1.6;
        margin-bottom: 16px;
    }
    .threshold-label {
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Chart theme constants ─────────────────────────────────────────────────────
_CHART_LAYOUT = dict(
    paper_bgcolor="#1E2538",
    plot_bgcolor="#1E2538",
    font=dict(color="#FFFFFF", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
)
_GREEN   = "#00D4A1"
_BLUE    = "#4A9EFF"
_RED     = "#FF4B6E"
_AMBER   = "#FFB347"

PRESSURE_COLORS = ["#4A9EFF", "#00D4A1", "#FFB347", "#FF4B6E"]  # low→extreme

# ── Pipeline loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading analytics pipeline...")
def load_pipeline():
    import warnings
    warnings.filterwarnings("ignore")
    from ranking_engine import run_full_pipeline
    return run_full_pipeline()


pipeline = load_pipeline()
full_df  = pipeline["full_df"]

# Import pressure engine constants once
try:
    from pressure_engine import (
        DEFAULT_BPI_WEIGHTS,
        DEFAULT_BOPI_WEIGHTS,
        PRESSURE_THRESHOLDS,
        compute_pressure,
    )
except Exception:
    DEFAULT_BPI_WEIGHTS = {
        "required_run_rate_difficulty": 0.30,
        "wickets_lost":                 0.20,
        "balls_remaining":              0.15,
        "innings_phase":                0.10,
        "match_leverage":               0.25,
    }
    DEFAULT_BOPI_WEIGHTS = {
        "opposition_scoring_threat": 0.25,
        "wickets_in_hand":           0.20,
        "balls_remaining":           0.15,
        "match_leverage":            0.25,
        "innings_phase":             0.15,
    }
    PRESSURE_THRESHOLDS = {
        "low":      (0,   25),
        "moderate": (25,  50),
        "high":     (50,  75),
        "extreme":  (75, 100),
    }
    compute_pressure = None

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:12px 0 16px 0;'>"
        "<span style='font-size:1.4rem; font-weight:900; color:#00D4A1;'>🏏 SUPRR</span>"
        "<div style='font-size:0.72rem; color:#A0ADB8; margin-top:2px;'>Pressure Analytics</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── BPI WEIGHTS ──
    st.markdown(
        "<div style='font-size:0.9rem; font-weight:700; color:#00D4A1; margin-bottom:6px;'>"
        "🔥 BPI Weights</div>",
        unsafe_allow_html=True,
    )
    st.caption("Weights are auto-normalized to sum to 1.0")

    bpi_rrr      = st.slider("Required RR Difficulty", 0.0, 1.0, DEFAULT_BPI_WEIGHTS["required_run_rate_difficulty"], 0.05, key="bpi_rrr")
    bpi_wickets  = st.slider("Wickets Lost",           0.0, 1.0, DEFAULT_BPI_WEIGHTS["wickets_lost"],                 0.05, key="bpi_wkts")
    bpi_balls    = st.slider("Balls Remaining",        0.0, 1.0, DEFAULT_BPI_WEIGHTS["balls_remaining"],              0.05, key="bpi_balls")
    bpi_phase    = st.slider("Innings Phase",          0.0, 1.0, DEFAULT_BPI_WEIGHTS["innings_phase"],               0.05, key="bpi_phase")
    bpi_leverage = st.slider("Match Leverage",         0.0, 1.0, DEFAULT_BPI_WEIGHTS["match_leverage"],              0.05, key="bpi_lev")

    bpi_total = bpi_rrr + bpi_wickets + bpi_balls + bpi_phase + bpi_leverage
    if abs(bpi_total) < 1e-9:
        bpi_total = 1.0  # avoid div-by-zero
    custom_bpi_weights = {
        "required_run_rate_difficulty": bpi_rrr     / bpi_total,
        "wickets_lost":                 bpi_wickets / bpi_total,
        "balls_remaining":              bpi_balls   / bpi_total,
        "innings_phase":                bpi_phase   / bpi_total,
        "match_leverage":               bpi_leverage / bpi_total,
    }

    st.markdown("---")

    # ── BOPI WEIGHTS ──
    st.markdown(
        "<div style='font-size:0.9rem; font-weight:700; color:#4A9EFF; margin-bottom:6px;'>"
        "🎳 BoPI Weights</div>",
        unsafe_allow_html=True,
    )
    st.caption("Weights are auto-normalized to sum to 1.0")

    bopi_threat   = st.slider("Opposition Scoring Threat", 0.0, 1.0, DEFAULT_BOPI_WEIGHTS["opposition_scoring_threat"], 0.05, key="bopi_threat")
    bopi_wih      = st.slider("Wickets in Hand",           0.0, 1.0, DEFAULT_BOPI_WEIGHTS["wickets_in_hand"],           0.05, key="bopi_wih")
    bopi_balls    = st.slider("Balls Remaining",           0.0, 1.0, DEFAULT_BOPI_WEIGHTS["balls_remaining"],           0.05, key="bopi_balls")
    bopi_leverage = st.slider("Match Leverage",            0.0, 1.0, DEFAULT_BOPI_WEIGHTS["match_leverage"],           0.05, key="bopi_lev")
    bopi_phase    = st.slider("Innings Phase",             0.0, 1.0, DEFAULT_BOPI_WEIGHTS["innings_phase"],            0.05, key="bopi_phase")

    bopi_total = bopi_threat + bopi_wih + bopi_balls + bopi_leverage + bopi_phase
    if abs(bopi_total) < 1e-9:
        bopi_total = 1.0
    custom_bopi_weights = {
        "opposition_scoring_threat": bopi_threat   / bopi_total,
        "wickets_in_hand":           bopi_wih      / bopi_total,
        "balls_remaining":           bopi_balls    / bopi_total,
        "match_leverage":            bopi_leverage / bopi_total,
        "innings_phase":             bopi_phase    / bopi_total,
    }

    st.markdown("---")

    # ── FILTERS ──
    st.markdown(
        "<div style='font-size:0.9rem; font-weight:700; color:#FFFFFF; margin-bottom:6px;'>"
        "🔍 Filters</div>",
        unsafe_allow_html=True,
    )
    fmt_filter    = st.selectbox("Format",  ["All", "IPL", "T20I"], key="fmt_filter")
    innings_filter = st.selectbox("Innings", ["All", "1st Innings", "2nd Innings"], key="inn_filter")
    phase_filter  = st.selectbox("Phase",   ["All", "Powerplay", "Middle", "Death"], key="phase_filter")

# ── Determine if weights have changed from defaults ───────────────────────────
def _weights_changed(custom: dict, defaults: dict) -> bool:
    for k in defaults:
        if abs(custom.get(k, 0) - defaults[k]) > 0.001:
            return True
    return False

bpi_changed  = _weights_changed(custom_bpi_weights,  DEFAULT_BPI_WEIGHTS)
bopi_changed = _weights_changed(custom_bopi_weights, DEFAULT_BOPI_WEIGHTS)
weights_changed = bpi_changed or bopi_changed


# ── Recompute pressure with custom weights (cached by weight tuple) ───────────
@st.cache_data(show_spinner=False)
def recompute_pressure(
    _base_df: pd.DataFrame,
    bpi_w_tuple: tuple,
    bopi_w_tuple: tuple,
) -> pd.DataFrame:
    """Recompute BPI/BoPI on the existing delivery dataframe with custom weights.
    Does NOT reload the pipeline — only recalculates pressure columns.
    """
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from pressure_engine import compute_pressure as _cp
        bpi_w  = dict(bpi_w_tuple)
        bopi_w = dict(bopi_w_tuple)
        result = _cp(_base_df, bpi_weights=bpi_w, bopi_weights=bopi_w)
        return result
    except Exception as exc:
        st.warning(f"Could not recompute pressure: {exc}")
        return _base_df


# Convert weight dicts to hashable tuples for cache key
bpi_key  = tuple(sorted(custom_bpi_weights.items()))
bopi_key = tuple(sorted(custom_bopi_weights.items()))

# Remove pressure columns from full_df so recompute works cleanly
_PRESSURE_COLS = [
    "bpi", "bopi", "pressure_category_bat", "pressure_category_bowl",
    "is_high_pressure_bat", "is_high_pressure_bowl",
    "bpi_rrr_component", "bpi_wickets_component", "bpi_balls_component",
    "bpi_phase_component", "bpi_leverage_component",
    "bopi_threat_component", "bopi_wih_component", "bopi_balls_component",
    "bopi_leverage_component", "bopi_phase_component",
]

if weights_changed and not full_df.empty:
    with st.spinner("Weights updated — re-running pressure calculation..."):
        # Strip old pressure cols so compute_pressure writes fresh ones
        base_for_recompute = full_df.drop(
            columns=[c for c in _PRESSURE_COLS if c in full_df.columns],
            errors="ignore",
        )
        working_df = recompute_pressure(base_for_recompute, bpi_key, bopi_key)
    st.success("Pressure recalculated with updated weights.", icon="✅")
else:
    working_df = full_df.copy()


# ── Apply sidebar filters ─────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    if fmt_filter != "All" and "match_format" in result.columns:
        result = result[result["match_format"].str.upper() == fmt_filter.upper()]
    if innings_filter != "All" and "innings" in result.columns:
        inn_num = 1 if innings_filter == "1st Innings" else 2
        result = result[result["innings"].astype(int) == inn_num]
    if phase_filter != "All" and "innings_phase" in result.columns:
        result = result[
            result["innings_phase"].str.lower().str.strip() == phase_filter.lower()
        ]
    return result


fdf = apply_filters(working_df)  # filtered dataframe for charts

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='padding: 32px 0 8px 0;'>"
    "<h1 style='font-size:2rem; font-weight:800; color:#FFFFFF; margin-bottom:4px;'>"
    "🔥 Pressure Detection Engine"
    "</h1>"
    "<div style='color:#A0ADB8; font-size:1rem;'>"
    "Explore BPI and BoPI distributions, phase patterns, and delivery-level pressure "
    "across formats. Adjust weights in the sidebar to explore sensitivity."
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<hr style='border-color:#1E2F4A; margin: 16px 0 20px 0;'>", unsafe_allow_html=True)

# Index explanation
st.markdown(
    "<div class='info-box'>"
    "<b style='color:#00D4A1;'>BPI — Batter Pressure Index (0–100):</b> "
    "Measures situational pressure on the batting side. Combines required run rate, "
    "wickets lost, balls remaining, innings phase, and match leverage. "
    "BPI ≥ 50 = high-pressure delivery for the batter."
    "<br><br>"
    "<b style='color:#4A9EFF;'>BoPI — Bowler Pressure Index (0–100):</b> "
    "Measures situational pressure on the bowling side. Combines opposition scoring "
    "threat, wickets in hand, balls remaining, match leverage, and innings phase. "
    "BoPI ≥ 50 = high-pressure delivery for the bowler."
    "</div>",
    unsafe_allow_html=True,
)

# Filter info
if any([fmt_filter != "All", innings_filter != "All", phase_filter != "All"]):
    active_filters = []
    if fmt_filter != "All":
        active_filters.append(f"Format: **{fmt_filter}**")
    if innings_filter != "All":
        active_filters.append(f"Innings: **{innings_filter}**")
    if phase_filter != "All":
        active_filters.append(f"Phase: **{phase_filter}**")
    st.info(f"Active filters — {' · '.join(active_filters)} · {len(fdf):,} deliveries shown")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BPI vs BoPI SUMMARY CARDS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📊 BPI vs BoPI Summary</div>", unsafe_allow_html=True)

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

if not fdf.empty and "bpi" in fdf.columns:
    mean_bpi        = fdf["bpi"].mean()
    high_bpi_count  = int(fdf["is_high_pressure_bat"].sum()) if "is_high_pressure_bat" in fdf.columns else int((fdf["bpi"] >= 50).sum())
    mean_bopi       = fdf["bopi"].mean() if "bopi" in fdf.columns else 0.0
    high_bopi_count = int(fdf["is_high_pressure_bowl"].sum()) if "is_high_pressure_bowl" in fdf.columns else int((fdf.get("bopi", pd.Series(dtype=float)) >= 50).sum())
else:
    mean_bpi = mean_bopi = 0.0
    high_bpi_count = high_bopi_count = 0

for col, label, value in [
    (kpi_c1, "Mean BPI",                      f"{mean_bpi:.1f}"),
    (kpi_c2, "High Pressure Deliveries (BPI)", f"{high_bpi_count:,}"),
    (kpi_c3, "Mean BoPI",                     f"{mean_bopi:.1f}"),
    (kpi_c4, "High Pressure Deliveries (BoPI)",f"{high_bopi_count:,}"),
]:
    with col:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value'>{value}</div>"
            f"<div class='metric-label'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PRESSURE DISTRIBUTION (BPI donut + BoPI bar)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🌡️ Pressure Category Distribution</div>", unsafe_allow_html=True)

col_bpi_pie, col_bopi_bar = st.columns(2, gap="large")

with col_bpi_pie:
    if not fdf.empty and "pressure_category_bat" in fdf.columns:
        bpi_cats = (
            fdf["pressure_category_bat"]
            .value_counts()
            .reindex(["low", "moderate", "high", "extreme"])
            .fillna(0)
            .reset_index()
        )
        bpi_cats.columns = ["Category", "Count"]

        fig_bpi_pie = go.Figure(
            go.Pie(
                labels=bpi_cats["Category"].str.capitalize(),
                values=bpi_cats["Count"],
                hole=0.52,
                marker=dict(colors=PRESSURE_COLORS, line=dict(color="#0A0F1E", width=2)),
                textinfo="label+percent",
                textfont=dict(color="#FFFFFF", size=12),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
            )
        )
        fig_bpi_pie.update_layout(
            title=dict(text="BPI Category Distribution", font=dict(color="#FFFFFF", size=14)),
            showlegend=True,
            legend=dict(font=dict(color="#FFFFFF")),
            **_CHART_LAYOUT,
        )
        st.plotly_chart(fig_bpi_pie, use_container_width=True)
    else:
        st.info("BPI category data not available.")

with col_bopi_bar:
    if not fdf.empty and "pressure_category_bowl" in fdf.columns:
        bopi_cats = (
            fdf["pressure_category_bowl"]
            .value_counts()
            .reindex(["low", "moderate", "high", "extreme"])
            .fillna(0)
            .reset_index()
        )
        bopi_cats.columns = ["Category", "Count"]

        fig_bopi_bar = go.Figure(
            go.Bar(
                x=bopi_cats["Category"].str.capitalize(),
                y=bopi_cats["Count"],
                marker=dict(color=PRESSURE_COLORS, line=dict(color="#0A0F1E", width=0.5)),
                hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
                text=bopi_cats["Count"].apply(lambda x: f"{int(x):,}"),
                textposition="outside",
                textfont=dict(color="#FFFFFF", size=10),
            )
        )
        fig_bopi_bar.update_layout(
            title=dict(text="BoPI Category Distribution", font=dict(color="#FFFFFF", size=14)),
            xaxis=dict(color="#A0ADB8"),
            yaxis=dict(title="Deliveries", color="#A0ADB8", gridcolor="#1E2F4A"),
            **_CHART_LAYOUT,
        )
        st.plotly_chart(fig_bopi_bar, use_container_width=True)
    else:
        st.info("BoPI category data not available.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PRESSURE BY PHASE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📐 Pressure by Phase</div>", unsafe_allow_html=True)

if not fdf.empty and "bpi" in fdf.columns and "innings_phase" in fdf.columns:
    phase_grp = (
        fdf.groupby(
            fdf["innings_phase"].str.lower().str.strip(), observed=True
        )[["bpi", "bopi"]]
        .mean()
        .reindex(["powerplay", "middle", "death"])
        .dropna(how="all")
        .reset_index()
    )
    phase_grp.columns = ["Phase", "Mean BPI", "Mean BoPI"]
    phase_grp["Phase"] = phase_grp["Phase"].str.capitalize()

    fig_phase = go.Figure()
    fig_phase.add_trace(go.Bar(
        name="Mean BPI",
        x=phase_grp["Phase"],
        y=phase_grp["Mean BPI"],
        marker_color=_GREEN,
        hovertemplate="<b>%{x}</b><br>Mean BPI: %{y:.2f}<extra></extra>",
        text=phase_grp["Mean BPI"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
        textposition="outside",
        textfont=dict(color=_GREEN, size=11),
    ))
    fig_phase.add_trace(go.Bar(
        name="Mean BoPI",
        x=phase_grp["Phase"],
        y=phase_grp["Mean BoPI"],
        marker_color=_BLUE,
        hovertemplate="<b>%{x}</b><br>Mean BoPI: %{y:.2f}<extra></extra>",
        text=phase_grp["Mean BoPI"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
        textposition="outside",
        textfont=dict(color=_BLUE, size=11),
    ))
    fig_phase.update_layout(
        title=dict(text="Mean BPI & BoPI by Innings Phase", font=dict(color="#FFFFFF", size=14)),
        barmode="group",
        xaxis=dict(color="#A0ADB8"),
        yaxis=dict(title="Mean Index Score", color="#A0ADB8", gridcolor="#1E2F4A", range=[0, 80]),
        legend=dict(font=dict(color="#FFFFFF")),
        **_CHART_LAYOUT,
    )
    st.plotly_chart(fig_phase, use_container_width=True)
else:
    st.info("Phase data not available for the current filter selection.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PRESSURE BY COMPETITION FORMAT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🌐 Pressure by Competition Format</div>", unsafe_allow_html=True)

if not fdf.empty and "bpi" in fdf.columns and "match_format" in fdf.columns:
    fmt_grp = (
        fdf.groupby(fdf["match_format"].str.upper(), observed=True)[["bpi", "bopi"]]
        .mean()
        .reset_index()
    )
    fmt_grp.columns = ["Format", "Mean BPI", "Mean BoPI"]
    # Keep only IPL and T20I rows
    fmt_grp = fmt_grp[fmt_grp["Format"].isin(["IPL", "T20I"])].copy()

    if not fmt_grp.empty:
        fig_fmt = go.Figure()
        fig_fmt.add_trace(go.Bar(
            name="Mean BPI",
            x=fmt_grp["Format"],
            y=fmt_grp["Mean BPI"],
            marker_color=_GREEN,
            hovertemplate="<b>%{x}</b><br>Mean BPI: %{y:.2f}<extra></extra>",
            text=fmt_grp["Mean BPI"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
            textposition="outside",
            textfont=dict(color=_GREEN, size=12),
        ))
        fig_fmt.add_trace(go.Bar(
            name="Mean BoPI",
            x=fmt_grp["Format"],
            y=fmt_grp["Mean BoPI"],
            marker_color=_BLUE,
            hovertemplate="<b>%{x}</b><br>Mean BoPI: %{y:.2f}<extra></extra>",
            text=fmt_grp["Mean BoPI"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
            textposition="outside",
            textfont=dict(color=_BLUE, size=12),
        ))
        fig_fmt.update_layout(
            title=dict(text="Mean BPI & BoPI — IPL vs T20I", font=dict(color="#FFFFFF", size=14)),
            barmode="group",
            xaxis=dict(color="#A0ADB8"),
            yaxis=dict(title="Mean Index Score", color="#A0ADB8", gridcolor="#1E2F4A", range=[0, 80]),
            legend=dict(font=dict(color="#FFFFFF")),
            **_CHART_LAYOUT,
        )
        st.plotly_chart(fig_fmt, use_container_width=True)
    else:
        st.info("IPL/T20I format split data not available.")
else:
    st.info("match_format column not found in data.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BPI COMPONENT CONTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🔩 BPI Component Contributions</div>", unsafe_allow_html=True)

bpi_component_cols = {
    "bpi_rrr_component":      "RRR Difficulty",
    "bpi_wickets_component":  "Wickets Lost",
    "bpi_balls_component":    "Balls Remaining",
    "bpi_phase_component":    "Innings Phase",
    "bpi_leverage_component": "Match Leverage",
}
available_bpi_comps = {k: v for k, v in bpi_component_cols.items() if k in fdf.columns}

if not fdf.empty and available_bpi_comps:
    comp_means = fdf[list(available_bpi_comps.keys())].mean().reset_index()
    comp_means.columns = ["Component", "Mean Score"]
    comp_means["Component"] = comp_means["Component"].map(available_bpi_comps)
    comp_means = comp_means.sort_values("Mean Score", ascending=True)

    fig_comp = go.Figure(
        go.Bar(
            x=comp_means["Mean Score"],
            y=comp_means["Component"],
            orientation="h",
            marker=dict(
                color=comp_means["Mean Score"],
                colorscale=[[0, _BLUE], [0.5, _GREEN], [1, _RED]],
                showscale=True,
                colorbar=dict(
                    title="Score",
                    tickfont=dict(color="#FFFFFF"),
                    titlefont=dict(color="#FFFFFF"),
                ),
            ),
            hovertemplate="<b>%{y}</b><br>Mean Component Score: %{x:.1f}<extra></extra>",
            text=comp_means["Mean Score"].apply(lambda x: f"{x:.1f}"),
            textposition="outside",
            textfont=dict(color="#FFFFFF", size=11),
        )
    )
    fig_comp.update_layout(
        title=dict(text="Mean BPI Component Scores (0–100 each)", font=dict(color="#FFFFFF", size=14)),
        xaxis=dict(title="Mean Score (0–100)", color="#A0ADB8", gridcolor="#1E2F4A", range=[0, 100]),
        yaxis=dict(color="#A0ADB8"),
        **_CHART_LAYOUT,
    )
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("BPI component columns not available in current data.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BPI vs BoPI SCATTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🔭 BPI vs BoPI Scatter</div>", unsafe_allow_html=True)

if not fdf.empty and "bpi" in fdf.columns and "bopi" in fdf.columns:
    # Sample up to 2000 points
    sample_size = min(2000, len(fdf))
    scatter_df = fdf.sample(n=sample_size, random_state=42).copy()

    if "innings_phase" in scatter_df.columns:
        scatter_df["Phase"] = scatter_df["innings_phase"].str.capitalize()
    else:
        scatter_df["Phase"] = "Unknown"

    phase_color_map = {
        "Powerplay": _GREEN,
        "Middle":    _BLUE,
        "Death":     _RED,
        "Unknown":   "#A0ADB8",
    }

    fig_scatter = go.Figure()
    for phase_label, color in phase_color_map.items():
        subset = scatter_df[scatter_df["Phase"] == phase_label]
        if subset.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=subset["bpi"],
            y=subset["bopi"],
            mode="markers",
            name=phase_label,
            marker=dict(
                color=color,
                size=5,
                opacity=0.55,
                line=dict(color="#0A0F1E", width=0.3),
            ),
            hovertemplate=(
                f"<b>{phase_label}</b><br>"
                "BPI: %{x:.1f}<br>BoPI: %{y:.1f}<extra></extra>"
            ),
        ))

    # Add diagonal reference line
    fig_scatter.add_shape(
        type="line",
        x0=0, y0=0, x1=100, y1=100,
        line=dict(color="#A0ADB8", width=1, dash="dot"),
    )
    fig_scatter.add_annotation(
        x=85, y=90,
        text="BPI = BoPI",
        showarrow=False,
        font=dict(color="#A0ADB8", size=10),
    )

    fig_scatter.update_layout(
        title=dict(
            text=f"BPI vs BoPI (sample of {sample_size:,} deliveries, coloured by phase)",
            font=dict(color="#FFFFFF", size=14),
        ),
        xaxis=dict(title="BPI (Batter Pressure)", color="#A0ADB8", gridcolor="#1E2F4A", range=[0, 100]),
        yaxis=dict(title="BoPI (Bowler Pressure)", color="#A0ADB8", gridcolor="#1E2F4A", range=[0, 100]),
        legend=dict(font=dict(color="#FFFFFF"), title=dict(text="Phase", font=dict(color="#FFFFFF"))),
        **_CHART_LAYOUT,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("BPI/BoPI columns not available in current data.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PRESSURE THRESHOLD TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📋 Pressure Threshold Reference</div>", unsafe_allow_html=True)

threshold_descriptions = {
    "low":      "Normal game state. Batter comfortable; bowler has little urgency.",
    "moderate": "Some situational pressure. The game is live; each ball matters.",
    "high":     "Clear pressure situation. Requires reliable execution under stress.",
    "extreme":  "Match-defining moment. Maximum pressure — results often decisive.",
}

if not fdf.empty and "pressure_category_bat" in fdf.columns:
    total_deliveries_in_view = len(fdf)
    cat_counts = fdf["pressure_category_bat"].value_counts()
else:
    total_deliveries_in_view = 0
    cat_counts = pd.Series(dtype=int)

rows = []
for label, (lo, hi) in PRESSURE_THRESHOLDS.items():
    count = int(cat_counts.get(label, 0))
    pct   = (count / total_deliveries_in_view * 100) if total_deliveries_in_view > 0 else 0.0
    rows.append({
        "BPI Range":    f"{lo}–{hi}",
        "Label":        label.capitalize(),
        "Description":  threshold_descriptions[label],
        "Count":        f"{count:,}",
        "% Deliveries": f"{pct:.1f}%",
    })

threshold_df = pd.DataFrame(rows)
st.dataframe(threshold_df, use_container_width=True, hide_index=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#A0ADB8; font-size:0.78rem; "
    "padding: 16px 0; border-top: 1px solid #1E2F4A;'>"
    "SUPRR Selector · Pressure Detection Engine · BPI &amp; BoPI Analysis"
    "</div>",
    unsafe_allow_html=True,
)
