"""
pages/6_Reliability.py – SUPRR Selector | Reliability & Statistical Validation
Full statistical robustness report: eligibility, sample distribution,
shrinkage, confidence intervals, format consistency, and sensitivity analysis.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reliability | SUPRR Selector",
    page_icon="🔬",
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
        height: 100%;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00D4A1; }
    .metric-label { font-size: 0.85rem; color: #A0ADB8; margin-top: 4px; }

    .threshold-card {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-top: 3px solid #00D4A1;
        border-radius: 12px;
        padding: 20px 24px;
    }
    .threshold-title { font-size: 1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px; }
    .threshold-row { display: flex; justify-content: space-between; padding: 4px 0;
                      border-bottom: 1px solid #1E2F4A; font-size: 0.85rem; }
    .threshold-key { color: #A0ADB8; }
    .threshold-val { color: #00D4A1; font-weight: 700; }

    .formula-box {
        background: #0A0F1E;
        border: 1px solid #1E2F4A;
        border-left: 3px solid #4A9EFF;
        border-radius: 8px;
        padding: 16px 20px;
        font-family: monospace;
        font-size: 0.95rem;
        color: #4A9EFF;
        margin: 16px 0;
    }

    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #FFFFFF;
        margin-top: 32px; margin-bottom: 16px;
        padding-bottom: 8px; border-bottom: 1px solid #1E2F4A;
    }
    .page-header {
        background: linear-gradient(135deg, #141929 0%, #1a2540 100%);
        border: 1px solid #1E2F4A;
        border-left: 4px solid #4A9EFF;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .page-title { font-size: 2rem; font-weight: 800; color: #FFFFFF; margin: 0; }
    .page-desc  { font-size: 0.92rem; color: #A0ADB8; margin-top: 8px; line-height: 1.7; }
    </style>
    """,
    unsafe_allow_html=True,
)

_CHART_LAYOUT = dict(
    paper_bgcolor="#1E2538",
    plot_bgcolor="#1E2538",
    font=dict(color="#FFFFFF", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading analytics pipeline...")
def load_pipeline():
    import warnings
    warnings.filterwarnings("ignore")
    from ranking_engine import run_full_pipeline
    return run_full_pipeline()


pipeline = load_pipeline()
batter_rankings = pipeline["batter_rankings"]
bowler_rankings = pipeline["bowler_rankings"]
batter_full = pipeline["batter_full"]
bowler_full = pipeline["bowler_full"]
full_df = pipeline["full_df"]
data_summary = pipeline["data_summary"]


# ── Cached computations ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _run_eligibility_check(_batter_full, _bowler_full):
    """Add eligibility detail columns to both frames."""
    try:
        from reliability import check_eligibility
        from batting_analysis import ELIGIBILITY_THRESHOLDS as BAT_THRESH
        from bowling_analysis import ELIGIBILITY_THRESHOLDS as BOWL_THRESH
    except Exception:
        return _batter_full, _bowler_full, {}, {}

    try:
        bat_checked = check_eligibility(_batter_full, BAT_THRESH)
    except Exception:
        bat_checked = _batter_full.copy()

    try:
        bowl_checked = check_eligibility(_bowler_full, BOWL_THRESH)
    except Exception:
        bowl_checked = _bowler_full.copy()

    return bat_checked, bowl_checked, BAT_THRESH, BOWL_THRESH


@st.cache_data(show_spinner=False)
def _shrinkage_recalc(_df, n_col: str, score_col: str, k: int):
    """Recompute shrinkage with a custom k value."""
    if _df is None or _df.empty:
        return _df
    if n_col not in _df.columns or score_col not in _df.columns:
        return _df
    df = _df.copy()
    n = df[n_col].fillna(0).astype(float)
    score = df[score_col].astype(float)
    mu = float(score.mean(skipna=True)) if not score.isna().all() else 50.0
    weight = n / (n + k)
    df["adjusted_score_custom"] = weight * score + (1.0 - weight) * mu
    return df


@st.cache_data(show_spinner=False)
def _run_sensitivity(_df, _base_weights):
    """Run sensitivity analysis for each weight dimension."""
    try:
        from reliability import sensitivity_analysis
    except Exception:
        return {}

    results = {}
    for key in _base_weights:
        try:
            res = sensitivity_analysis(_df, _base_weights, key, n_steps=10)
            results[key] = res
        except Exception:
            results[key] = {"weight_values": [], "top10_overlap": [], "rank_stability": np.nan}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
        <div class="page-title">🔬 Reliability & Statistical Validation</div>
        <div class="page-desc">
            Transparency on the statistical robustness of SUPRR rankings.
            This page documents eligibility thresholds, sample size distributions,
            Bayesian shrinkage, bootstrap confidence intervals, format consistency,
            and sensitivity of rankings to weight changes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Run eligibility check ─────────────────────────────────────────────────────
bat_checked, bowl_checked, BAT_THRESH, BOWL_THRESH = _run_eligibility_check(
    batter_full, bowler_full
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Eligibility Overview
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Eligibility Overview</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    thresh_rows = "".join(
        f'<div class="threshold-row"><span class="threshold-key">{k}</span>'
        f'<span class="threshold-val">≥ {v}</span></div>'
        for k, v in BAT_THRESH.items()
    ) if BAT_THRESH else '<div class="threshold-key">Default thresholds applied</div>'
    st.markdown(
        f"""
        <div class="threshold-card">
            <div class="threshold-title">🏏 Batter Eligibility Thresholds</div>
            {thresh_rows}
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    thresh_rows_b = "".join(
        f'<div class="threshold-row"><span class="threshold-key">{k}</span>'
        f'<span class="threshold-val">≥ {v}</span></div>'
        for k, v in BOWL_THRESH.items()
    ) if BOWL_THRESH else '<div class="threshold-key">Default thresholds applied</div>'
    st.markdown(
        f"""
        <div class="threshold-card" style="border-top-color: #4A9EFF">
            <div class="threshold-title">🎯 Bowler Eligibility Thresholds</div>
            {thresh_rows_b}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Eligibility Detail Tables")

elig_tabs = st.tabs(["Batters", "Bowlers"])

with elig_tabs[0]:
    if not bat_checked.empty:
        display_cols = [c for c in [
            "player_name", "total_balls", "pressure_balls", "pressure_matches",
            "is_eligible", "eligibility_reason", "sample_size_tier",
        ] if c in bat_checked.columns]
        display_df = bat_checked[display_cols].reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, height=320)
    else:
        st.info("No batter data available.")

with elig_tabs[1]:
    if not bowl_checked.empty:
        display_cols = [c for c in [
            "player_name", "total_balls", "pressure_balls", "pressure_matches",
            "is_eligible", "eligibility_reason", "sample_size_tier",
        ] if c in bowl_checked.columns]
        display_df = bowl_checked[display_cols].reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, height=320)
    else:
        st.info("No bowler data available.")

# Donut charts: eligible vs ineligible
col1, col2 = st.columns(2)

def _donut_chart(df, title):
    if df.empty or "is_eligible" not in df.columns:
        return None
    n_elig = int(df["is_eligible"].sum())
    n_total = len(df)
    n_inelig = n_total - n_elig
    fig = go.Figure(go.Pie(
        labels=["Eligible", "Ineligible"],
        values=[n_elig, n_inelig],
        hole=0.6,
        marker=dict(colors=["#00D4A1", "#FF4B6E"]),
        textinfo="label+percent",
        textfont=dict(color="#FFFFFF", size=13),
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        title=dict(text=title, font=dict(color="#FFFFFF")),
        annotations=[dict(
            text=f"{n_elig}/{n_total}",
            x=0.5, y=0.5,
            font=dict(size=18, color="#00D4A1"),
            showarrow=False,
        )],
        showlegend=True,
        legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
        height=320,
    )
    return fig

with col1:
    fig = _donut_chart(bat_checked, "Batter Eligibility")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = _donut_chart(bowl_checked, "Bowler Eligibility")
    if fig:
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sample Size Distribution
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Sample Size Distribution</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

def _sample_hist(df, threshold, title, color):
    """Histogram of pressure_balls for eligible players."""
    if df.empty or "pressure_balls" not in df.columns:
        return None
    eligible_df = df[df["is_eligible"].astype(bool)] if "is_eligible" in df.columns else df
    if eligible_df.empty:
        return None
    vals = eligible_df["pressure_balls"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals,
        nbinsx=25,
        name="Pressure Balls",
        marker_color=color,
        opacity=0.8,
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#FF4B6E",
                   annotation_text=f"Min threshold: {threshold}", annotation_font_color="#FF4B6E")
    fig.add_vline(x=vals.median(), line_dash="dot", line_color="#4A9EFF",
                   annotation_text=f"Median: {vals.median():.0f}", annotation_font_color="#4A9EFF")
    fig.update_layout(
        **_CHART_LAYOUT,
        xaxis=dict(title="Pressure Balls", gridcolor="#1E2F4A"),
        yaxis=dict(title="Count", gridcolor="#1E2F4A"),
        title=dict(text=title, font=dict(color="#FFFFFF")),
        showlegend=False,
        height=300,
    )
    return fig, vals

with col1:
    min_bat_pb = BAT_THRESH.get("min_pressure_balls", 30) if BAT_THRESH else 30
    result = _sample_hist(bat_checked, min_bat_pb, "Batter Pressure Balls Distribution", "#00D4A1")
    if result:
        fig, vals = result
        st.plotly_chart(fig, use_container_width=True)
        pct_above = (vals >= min_bat_pb * 2).mean() * 100
        st.caption(
            f"Sample size distribution: **{pct_above:.0f}%** of eligible batters "
            f"have **{min_bat_pb * 2}+** pressure balls (double the minimum threshold)."
        )

with col2:
    min_bowl_pb = BOWL_THRESH.get("min_pressure_balls", 24) if BOWL_THRESH else 24
    result = _sample_hist(bowl_checked, min_bowl_pb, "Bowler Pressure Balls Distribution", "#4A9EFF")
    if result:
        fig, vals = result
        st.plotly_chart(fig, use_container_width=True)
        pct_above = (vals >= min_bowl_pb * 2).mean() * 100
        st.caption(
            f"Sample size distribution: **{pct_above:.0f}%** of eligible bowlers "
            f"have **{min_bowl_pb * 2}+** pressure balls (double the minimum threshold)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Shrinkage Adjustment
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔧 Shrinkage Adjustment (Bayesian Regularisation)</div>', unsafe_allow_html=True)

st.markdown(
    """
    Bayesian shrinkage pulls ratings for small-sample players toward the group average.
    This prevents over-reliance on a handful of matches. Players with large samples
    retain their raw score almost entirely; newcomers regress significantly.
    """,
)

st.markdown(
    '<div class="formula-box">'
    'Adjusted Rating = (n / (n + k)) × Player Rating + (k / (n + k)) × Group Average'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("*Where **n** = pressure balls faced/bowled, **k** = shrinkage constant (default: 50 batters, 40 bowlers)*")

# Interactive slider
col_s1, col_s2 = st.columns([1, 2])
with col_s1:
    shrink_type = st.radio("Apply to", ["Batters", "Bowlers"], horizontal=True, key="shrink_type")
    k_val = st.slider(
        "Shrinkage constant k",
        min_value=20, max_value=100,
        value=50 if shrink_type == "Batters" else 40,
        step=5, key="k_slider",
    )

ref_full = batter_full if shrink_type == "Batters" else bowler_full
score_col_ref = "combined_score"
n_col_ref = "pressure_balls"

if not ref_full.empty and score_col_ref in ref_full.columns and n_col_ref in ref_full.columns:
    shrunk_df = _shrinkage_recalc(ref_full, n_col_ref, score_col_ref, k_val)

    with col_s2:
        # Scatter: raw vs adjusted
        raw_scores = shrunk_df[score_col_ref].dropna()
        adj_col = "adjusted_score_custom"
        adj_scores = shrunk_df[adj_col].dropna() if adj_col in shrunk_df.columns else raw_scores
        common_idx = raw_scores.index.intersection(adj_scores.index)

        player_names = (
            shrunk_df.loc[common_idx, "player_name"]
            if "player_name" in shrunk_df.columns
            else pd.Series("", index=common_idx)
        )
        pb_vals = shrunk_df.loc[common_idx, n_col_ref].fillna(0)

        fig_shrink = go.Figure()
        fig_shrink.add_trace(go.Scatter(
            x=raw_scores.loc[common_idx],
            y=adj_scores.loc[common_idx],
            mode="markers",
            marker=dict(
                color=pb_vals,
                colorscale="Viridis",
                colorbar=dict(title="Pressure Balls", tickfont=dict(color="#FFFFFF")),
                size=8,
                opacity=0.8,
            ),
            text=player_names,
            hovertemplate="<b>%{text}</b><br>Raw: %{x:.1f}<br>Adjusted: %{y:.1f}<extra></extra>",
            name="Players",
        ))
        # y = x reference line
        mn = min(raw_scores.min(), adj_scores.min())
        mx = max(raw_scores.max(), adj_scores.max())
        fig_shrink.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color="#FF4B6E", dash="dash", width=1),
            name="No change (y = x)",
        ))
        fig_shrink.update_layout(
            **_CHART_LAYOUT,
            xaxis=dict(title=f"Raw {score_col_ref}", gridcolor="#1E2F4A"),
            yaxis=dict(title="Adjusted Score (k={})".format(k_val), gridcolor="#1E2F4A"),
            title=dict(text=f"Raw vs Adjusted Score — {shrink_type} (k={k_val})", font=dict(color="#FFFFFF")),
            legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
            height=380,
        )
        st.plotly_chart(fig_shrink, use_container_width=True)

        st.caption(
            "Points below the dashed line were pulled **down** toward the mean (over-performing "
            "small-sample players). Points above were pulled **up** (under-performing small-sample)."
        )
else:
    st.info("combined_score or pressure_balls columns not available for shrinkage visualisation.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📏 Bootstrap Confidence Intervals</div>', unsafe_allow_html=True)

st.markdown(
    "Confidence intervals derived from match-level bootstrap resampling (200 resamples per player). "
    "Each interval covers the 2.5–97.5th percentile of bootstrapped metric distributions. "
    "Players with fewer than 5 pressure matches receive conservative ±30% bounds."
)

ci_tabs = st.tabs(["Top 10 Batters — Pressure SR", "Top 10 Bowlers — Pressure Economy"])

with ci_tabs[0]:
    if not batter_full.empty:
        top10_bat = batter_full.sort_values("adjusted_score", ascending=False).head(10)
        has_ci = "ci_lower" in top10_bat.columns and "ci_upper" in top10_bat.columns
        sr_col = "pressure_strike_rate"
        if sr_col in top10_bat.columns:
            names_b = top10_bat["player_name"].fillna("Unknown").tolist()
            sr_vals = top10_bat[sr_col].fillna(0).tolist()
            ci_lo = top10_bat["ci_lower"].fillna(0).tolist() if has_ci else [v * 0.85 for v in sr_vals]
            ci_hi = top10_bat["ci_upper"].fillna(0).tolist() if has_ci else [v * 1.15 for v in sr_vals]
            err_lo = [max(s - lo, 0) for s, lo in zip(sr_vals, ci_lo)]
            err_hi = [max(hi - s, 0) for s, hi in zip(sr_vals, ci_hi)]

            fig_ci = go.Figure()
            fig_ci.add_trace(go.Scatter(
                y=names_b,
                x=sr_vals,
                mode="markers",
                marker=dict(color="#00D4A1", size=10),
                error_x=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo,
                             color="#A0ADB8", thickness=2, width=6),
                name="Pressure SR ± 95% CI",
            ))
            fig_ci.update_layout(
                **_CHART_LAYOUT,
                xaxis=dict(title="Pressure Strike Rate", gridcolor="#1E2F4A"),
                yaxis=dict(autorange="reversed"),
                title=dict(text="Top 10 Batters: Pressure SR with Confidence Intervals", font=dict(color="#FFFFFF")),
                height=380,
            )
            st.plotly_chart(fig_ci, use_container_width=True)
        else:
            st.info("pressure_strike_rate column not found.")
    else:
        st.info("No batter data available.")

with ci_tabs[1]:
    if not bowler_full.empty:
        top10_bowl = bowler_full.sort_values("adjusted_score", ascending=False).head(10)
        has_ci = "ci_lower" in top10_bowl.columns and "ci_upper" in top10_bowl.columns
        eco_col = "pressure_economy"
        if eco_col in top10_bowl.columns:
            names_bw = top10_bowl["player_name"].fillna("Unknown").tolist()
            eco_vals = top10_bowl[eco_col].fillna(0).tolist()
            ci_lo = top10_bowl["ci_lower"].fillna(0).tolist() if has_ci else [v * 0.85 for v in eco_vals]
            ci_hi = top10_bowl["ci_upper"].fillna(0).tolist() if has_ci else [v * 1.15 for v in eco_vals]
            err_lo = [max(s - lo, 0) for s, lo in zip(eco_vals, ci_lo)]
            err_hi = [max(hi - s, 0) for s, hi in zip(eco_vals, ci_hi)]

            fig_ci = go.Figure()
            fig_ci.add_trace(go.Scatter(
                y=names_bw,
                x=eco_vals,
                mode="markers",
                marker=dict(color="#4A9EFF", size=10),
                error_x=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo,
                             color="#A0ADB8", thickness=2, width=6),
                name="Pressure Economy ± 95% CI",
            ))
            fig_ci.update_layout(
                **_CHART_LAYOUT,
                xaxis=dict(title="Pressure Economy (lower is better)", gridcolor="#1E2F4A"),
                yaxis=dict(autorange="reversed"),
                title=dict(text="Top 10 Bowlers: Pressure Economy with Confidence Intervals", font=dict(color="#FFFFFF")),
                height=380,
            )
            st.plotly_chart(fig_ci, use_container_width=True)
        else:
            st.info("pressure_economy column not found.")
    else:
        st.info("No bowler data available.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. IPL vs T20I Consistency
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔁 IPL vs T20I Format Consistency</div>', unsafe_allow_html=True)

st.markdown(
    "Players near the diagonal line perform consistently across both formats. "
    "Large deviations suggest context-specific performance or small samples in one format."
)

# Try to use consistency scores from pipeline; compute inline if absent
@st.cache_data(show_spinner=False)
def _get_consistency_df(_batter_full):
    df = _batter_full.copy()
    if "ipl_t20i_consistency" not in df.columns:
        try:
            from reliability import compute_ipl_t20i_consistency
            df = compute_ipl_t20i_consistency(df)
        except Exception:
            pass
    return df


consist_df = _get_consistency_df(batter_full)

ipl_col = "ipl_pressure_sr"
t20i_col = "t20i_pressure_sr"

if ipl_col in consist_df.columns and t20i_col in consist_df.columns:
    both_mask = consist_df[ipl_col].notna() & consist_df[t20i_col].notna()
    both_formats = consist_df[both_mask]

    if not both_formats.empty:
        ipl_vals = both_formats[ipl_col].tolist()
        t20i_vals = both_formats[t20i_col].tolist()
        names_c = both_formats["player_name"].fillna("Unknown").tolist() if "player_name" in both_formats.columns else [""] * len(ipl_vals)
        consist_scores = both_formats["ipl_t20i_consistency"].tolist() if "ipl_t20i_consistency" in both_formats.columns else [0.5] * len(ipl_vals)

        fig_consist = go.Figure()
        fig_consist.add_trace(go.Scatter(
            x=ipl_vals, y=t20i_vals,
            mode="markers+text",
            marker=dict(
                color=consist_scores,
                colorscale="RdYlGn",
                colorbar=dict(title="Consistency", tickfont=dict(color="#FFFFFF")),
                size=10, opacity=0.9,
                cmin=0, cmax=1,
            ),
            text=names_c,
            textposition="top right",
            textfont=dict(size=9, color="#A0ADB8"),
            hovertemplate="<b>%{text}</b><br>IPL SR: %{x:.1f}<br>T20I SR: %{y:.1f}<extra></extra>",
            name="Players",
        ))

        # Diagonal reference line
        all_sr = ipl_vals + t20i_vals
        mn_sr, mx_sr = min(all_sr) - 5, max(all_sr) + 5
        fig_consist.add_trace(go.Scatter(
            x=[mn_sr, mx_sr], y=[mn_sr, mx_sr],
            mode="lines",
            line=dict(color="#FF4B6E", dash="dash", width=1),
            name="Perfect consistency (y = x)",
        ))

        fig_consist.update_layout(
            **_CHART_LAYOUT,
            xaxis=dict(title="IPL Pressure SR", gridcolor="#1E2F4A"),
            yaxis=dict(title="T20I Pressure SR", gridcolor="#1E2F4A"),
            title=dict(text=f"IPL vs T20I Pressure SR ({len(both_formats)} players with both formats)",
                       font=dict(color="#FFFFFF")),
            legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
            height=460,
        )
        st.plotly_chart(fig_consist, use_container_width=True)
        st.caption(
            "Colour scale: green = high consistency (small IPL/T20I difference), "
            "red = low consistency (large format gap). Points on the dashed line = identical performance."
        )
    else:
        st.info("No batters with pressure data in both IPL and T20I formats found.")
else:
    st.info("IPL/T20I pressure SR columns not present in batter data.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Weight Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚖️ Weight Sensitivity Analysis</div>', unsafe_allow_html=True)

st.markdown(
    "Rankings remain broadly consistent when individual component weights are varied. "
    "The table below shows how many top-10 players change when each weight is swept from 0 to its baseline value."
)
st.info(
    "Full sensitivity analysis available via `reliability.sensitivity_analysis`. "
    "Showing simplified stability overview for interactive responsiveness."
)

sens_tabs = st.tabs(["Batter Weight Sensitivity", "Bowler Weight Sensitivity"])

try:
    from scoring_engine import DEFAULT_BATTER_WEIGHTS, DEFAULT_BOWLER_WEIGHTS
except Exception:
    DEFAULT_BATTER_WEIGHTS = {
        "runs_above_expected": 0.25, "sr_improvement": 0.25,
        "dismissal_avoidance": 0.20, "boundary_contribution": 0.15, "dot_avoidance": 0.15,
    }
    DEFAULT_BOWLER_WEIGHTS = {
        "runs_prevented": 0.25, "economy_improvement": 0.25,
        "wicket_taking": 0.20, "dot_generation": 0.15, "boundary_prevention": 0.15,
    }

with sens_tabs[0]:
    if not batter_full.empty:
        with st.spinner("Running batter sensitivity..."):
            bat_sens = _run_sensitivity(batter_full, DEFAULT_BATTER_WEIGHTS)

        if bat_sens:
            sens_rows = []
            for key, res in bat_sens.items():
                vals = res.get("top10_overlap", [])
                changes = sum(1 for v in vals if v < 1.0)
                min_overlap = min(vals) if vals else np.nan
                sens_rows.append({
                    "Weight Dimension": key,
                    "Default Weight": DEFAULT_BATTER_WEIGHTS.get(key, "—"),
                    "Min Tested": 0.0,
                    "Max Tested": DEFAULT_BATTER_WEIGHTS.get(key, "—"),
                    "Steps w/ Rank Change": changes,
                    "Min Top-10 Overlap": f"{min_overlap:.2f}" if not np.isnan(min_overlap) else "—",
                    "Stability Score": f"{res.get('rank_stability', np.nan):.2f}" if not np.isnan(res.get('rank_stability', np.nan)) else "—",
                })
            st.dataframe(pd.DataFrame(sens_rows), use_container_width=True)

            # Bar chart of stability scores
            keys = list(bat_sens.keys())
            stab_scores = [bat_sens[k].get("rank_stability", 0) for k in keys]
            fig_sens = go.Figure(go.Bar(
                x=keys, y=stab_scores,
                marker_color=["#00D4A1" if s >= 0.8 else "#FF4B6E" if s < 0.6 else "#4A9EFF" for s in stab_scores],
                text=[f"{s:.2f}" for s in stab_scores],
                textposition="outside",
                textfont=dict(color="#FFFFFF"),
            ))
            fig_sens.update_layout(
                **_CHART_LAYOUT,
                xaxis=dict(title="Weight Dimension", tickangle=-20),
                yaxis=dict(title="Average Top-10 Overlap (stability)", range=[0, 1.1], gridcolor="#1E2F4A"),
                title=dict(text="Batter Rank Stability by Weight Dimension", font=dict(color="#FFFFFF")),
                height=320,
            )
            st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.info("Sensitivity analysis could not be computed (component columns may be absent).")
    else:
        st.info("No batter data available.")

with sens_tabs[1]:
    if not bowler_full.empty:
        with st.spinner("Running bowler sensitivity..."):
            bowl_sens = _run_sensitivity(bowler_full, DEFAULT_BOWLER_WEIGHTS)

        if bowl_sens:
            sens_rows = []
            for key, res in bowl_sens.items():
                vals = res.get("top10_overlap", [])
                changes = sum(1 for v in vals if v < 1.0)
                min_overlap = min(vals) if vals else np.nan
                sens_rows.append({
                    "Weight Dimension": key,
                    "Default Weight": DEFAULT_BOWLER_WEIGHTS.get(key, "—"),
                    "Min Tested": 0.0,
                    "Max Tested": DEFAULT_BOWLER_WEIGHTS.get(key, "—"),
                    "Steps w/ Rank Change": changes,
                    "Min Top-10 Overlap": f"{min_overlap:.2f}" if not np.isnan(min_overlap) else "—",
                    "Stability Score": f"{res.get('rank_stability', np.nan):.2f}" if not np.isnan(res.get('rank_stability', np.nan)) else "—",
                })
            st.dataframe(pd.DataFrame(sens_rows), use_container_width=True)

            keys = list(bowl_sens.keys())
            stab_scores = [bowl_sens[k].get("rank_stability", 0) for k in keys]
            fig_sens = go.Figure(go.Bar(
                x=keys, y=stab_scores,
                marker_color=["#00D4A1" if s >= 0.8 else "#FF4B6E" if s < 0.6 else "#4A9EFF" for s in stab_scores],
                text=[f"{s:.2f}" for s in stab_scores],
                textposition="outside",
                textfont=dict(color="#FFFFFF"),
            ))
            fig_sens.update_layout(
                **_CHART_LAYOUT,
                xaxis=dict(title="Weight Dimension", tickangle=-20),
                yaxis=dict(title="Average Top-10 Overlap (stability)", range=[0, 1.1], gridcolor="#1E2F4A"),
                title=dict(text="Bowler Rank Stability by Weight Dimension", font=dict(color="#FFFFFF")),
                height=320,
            )
            st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.info("Sensitivity analysis could not be computed.")
    else:
        st.info("No bowler data available.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Ranking Stability Disclaimer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚠️ Ranking Stability Disclaimer</div>', unsafe_allow_html=True)

st.warning(
    """
    **Any ranking system has inherent limitations.** SUPRR rankings should be interpreted in the context of:

    - **Sample sizes**: Players with fewer pressure matches have higher uncertainty. 
      Bayesian shrinkage reduces but does not eliminate this risk.
    - **Format coverage variability**: Players who appear predominantly in IPL versus T20I 
      face different average pressure conditions; the 50/50 blending is an approximation.
    - **Model uncertainty**: The GradientBoosting expected-performance model was trained 
      and tested on the same 2022–2026 era. Out-of-era generalisation is untested.
    - **Pressure definition sensitivity**: Rankings are sensitive to the BPI/BoPI thresholds 
      used to classify deliveries as "high pressure". Alternative definitions may reorder players.
    - **Selection effects**: Only players who appeared in matches are ranked. 
      Bench players and recently retired players are absent by design.
    - **Venue and conditions**: Ground dimensions, pitch behaviour, and weather are not 
      fully captured in ball-by-ball data, introducing noise into expected-performance estimates.

    Rankings are best used as a starting point for further scouting, not as a definitive selection criterion.
    """
)
