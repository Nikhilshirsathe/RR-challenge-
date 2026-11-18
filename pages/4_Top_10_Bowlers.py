"""
pages/4_Top_10_Bowlers.py – SUPRR Selector | Top 10 Pressure Bowlers
Full interactive rankings, radar charts, player profiles,
and statistical justification for the best pressure bowlers.
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
    page_title="Top 10 Bowlers | SUPRR Selector",
    page_icon="🎳",
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
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #4A9EFF; }
    .metric-label { font-size: 0.85rem; color: #A0ADB8; margin-top: 4px; }
    .metric-sub   { font-size: 0.78rem; color: #00D4A1; margin-top: 2px; }

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

    .profile-header {
        background: linear-gradient(135deg, #141929, #1a2a4a);
        border: 1px solid #1E2F4A;
        border-top: 3px solid #4A9EFF;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .profile-name  { font-size: 1.6rem; font-weight: 800; color: #FFFFFF; }
    .profile-team  { font-size: 0.9rem; color: #A0ADB8; margin-top: 2px; }
    .rank-pill {
        display: inline-block;
        background: linear-gradient(135deg, #4A9EFF, #00D4A1);
        color: #0A0F1E;
        font-weight: 800;
        font-size: 0.9rem;
        padding: 4px 14px;
        border-radius: 20px;
        margin-top: 8px;
    }
    .stat-card {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #4A9EFF; }
    .stat-label { font-size: 0.78rem; color: #A0ADB8; margin-top: 2px; }

    .justify-box {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-left: 4px solid #00D4A1;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 0.9rem;
        color: #A0ADB8;
        line-height: 1.8;
        margin-top: 12px;
    }
    .ci-box {
        background: #0d1527;
        border: 1px solid #1E2F4A;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.84rem;
        color: #A0ADB8;
        margin-top: 8px;
    }
    .format-mini {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .format-title { font-size: 0.8rem; font-weight: 700; color: #4A9EFF; text-transform: uppercase; letter-spacing: 0.08em; }
    .format-val   { font-size: 1.3rem; font-weight: 700; color: #4A9EFF; }
    .format-sub   { font-size: 0.75rem; color: #A0ADB8; }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Chart theme ───────────────────────────────────────────────────────────────
_CL = dict(
    paper_bgcolor="#1E2538",
    plot_bgcolor="#1E2538",
    font=dict(color="#FFFFFF", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)
_GREEN = "#00D4A1"
_BLUE  = "#4A9EFF"
_RED   = "#FF4B6E"

# ── Pipeline loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading analytics pipeline...")
def load_pipeline():
    import warnings
    warnings.filterwarnings("ignore")
    from ranking_engine import run_full_pipeline
    return run_full_pipeline()

pipeline        = load_pipeline()
batter_rankings = pipeline["batter_rankings"]
bowler_rankings = pipeline["bowler_rankings"]
batter_full     = pipeline["batter_full"]
bowler_full     = pipeline["bowler_full"]
full_df         = pipeline["full_df"]

# ── Helper: safe value ────────────────────────────────────────────────────────
def _v(val, fmt=None, prefix="", suffix="", fallback="N/A"):
    if val is None:
        return fallback
    try:
        f = float(val)
        if np.isnan(f):
            return fallback
        if fmt == "1f":
            s = f"{f:.1f}"
        elif fmt == "2f":
            s = f"{f:.2f}"
        elif fmt == "+1f":
            s = f"{f:+.1f}"
        elif fmt == "pct":
            s = f"{f:.1f}%"
        elif fmt == "int":
            s = str(int(round(f)))
        else:
            s = str(val)
        return f"{prefix}{s}{suffix}"
    except (TypeError, ValueError):
        return str(val) if val else fallback

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎳 Bowler Filters")
    st.markdown("---")

    fmt_choice = st.selectbox(
        "Format",
        ["All Formats", "IPL Only", "T20I Only"],
        index=0,
    )

    min_balls = st.slider("Minimum Pressure Balls", 10, 150, 24, 2)
    top_n = st.selectbox("Show Top N", [5, 10, 15, 20], index=1)

    st.markdown("---")

    @st.cache_data(show_spinner=False)
    def _download_df(_bf, _min_balls):
        df = _bf.copy()
        if "pressure_balls" in df.columns:
            df = df[df["pressure_balls"] >= _min_balls]
        return df.to_csv(index=False).encode("utf-8")

    dl_csv = _download_df(bowler_full, min_balls)
    st.download_button(
        label="⬇ Download Rankings CSV",
        data=dl_csv,
        file_name="suprr_bowler_rankings.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── Filter & sort data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def filter_bowlers(_df_full, fmt, min_b):
    df = _df_full.copy()
    if "pressure_balls" in df.columns:
        df = df[df["pressure_balls"] >= min_b]

    if fmt == "IPL Only":
        sort_col = "ipl_pressure_economy" if "ipl_pressure_economy" in df.columns else "adjusted_score"
        if "ipl_pressure_balls" in df.columns:
            df = df[pd.notna(df["ipl_pressure_balls"]) & (df["ipl_pressure_balls"] > 0)]
        df = df.sort_values(sort_col, ascending=True)   # lower economy = better
    elif fmt == "T20I Only":
        sort_col = "t20i_pressure_economy" if "t20i_pressure_economy" in df.columns else "adjusted_score"
        if "t20i_pressure_balls" in df.columns:
            df = df[pd.notna(df["t20i_pressure_balls"]) & (df["t20i_pressure_balls"] > 0)]
        df = df.sort_values(sort_col, ascending=True)
    else:
        df = df.sort_values("adjusted_score", ascending=False)

    df = df.reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df

filtered_full = filter_bowlers(bowler_full, fmt_choice, min_balls)
top_df        = filtered_full.head(top_n).copy()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
      <div class="page-title">🎳 Top Pressure Bowlers — SUPRR Rankings</div>
      <div class="page-desc">
        Rankings use the SUPRR (Situational Under-Pressure Performance Rating) model,
        evaluating bowlers when the Ball-Over Pressure Index (BoPI) signals high-pressure
        overs. Each adjusted score weights economy improvement, runs prevented above
        expectation, wicket-taking ability, dot ball generation, and boundary prevention
        — with Bayesian shrinkage to guard against small-sample extremes.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Warnings ──────────────────────────────────────────────────────────────────
if len(filtered_full) == 0:
    st.warning("No bowlers meet the current filter criteria. Try reducing the minimum pressure balls.")
    st.stop()
elif len(filtered_full) < top_n:
    st.info(
        f"Only **{len(filtered_full)}** bowlers meet the current filters "
        f"(minimum {min_balls} pressure balls, {fmt_choice}). Showing all eligible players."
    )

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Summary</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

eligible_count = len(filtered_full)
top10_slice    = filtered_full.head(10)

avg_econ  = top10_slice["pressure_economy"].mean()     if "pressure_economy" in top10_slice else np.nan
avg_imp   = top10_slice["economy_improvement"].mean()  if "economy_improvement" in top10_slice else np.nan
total_wk  = top10_slice["pressure_wickets"].sum()      if "pressure_wickets" in top10_slice else np.nan

for col, label, val, fmt, sub in [
    (k1, "Eligible Bowlers Analysed", eligible_count, "int",  fmt_choice),
    (k2, "Avg Pressure Economy",      avg_econ,       "1f",   "Top 10 players"),
    (k3, "Avg Economy Improvement",   avg_imp,        "+1f",  "vs. baseline"),
    (k4, "Total Pressure Wickets",    total_wk,       "int",  "Top 10 combined"),
]:
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-value">{_v(val, fmt)}</div>
              <div class="metric-label">{label}</div>
              <div class="metric-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Rankings Table ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Top N Rankings</div>', unsafe_allow_html=True)

def _build_display_table(df):
    cols_map = {
        "rank":                  "Rank",
        "player_name":           "Player",
        "main_team_name":        "Team",
        "adjusted_score":        "Pressure Rating",
        "pressure_economy":      "Pressure Economy",
        "baseline_economy":      "Baseline Economy",
        "economy_improvement":   "Economy Impr.",
        "pressure_wickets":      "Wickets",
        "wickets_above_expected":"Wkts Above Exp",
        "pressure_balls":        "Pressure Balls",
        "pressure_matches":      "Matches",
        "format_coverage":       "Format Coverage",
    }
    available = {k: v for k, v in cols_map.items() if k in df.columns}
    out = df[list(available.keys())].copy()
    out.columns = list(available.values())

    if "Economy Impr." in out.columns:
        out["Economy Impr."] = out["Economy Impr."].apply(
            lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A"
        )
    for c in ["Pressure Economy", "Baseline Economy"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    if "Pressure Rating" in out.columns:
        out["Pressure Rating"] = out["Pressure Rating"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
    if "Wkts Above Exp" in out.columns:
        out["Wkts Above Exp"] = out["Wkts Above Exp"].apply(
            lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A"
        )
    for c in ["Wickets", "Pressure Balls"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: str(int(round(x))) if pd.notna(x) else "N/A")
    return out

display_tbl = _build_display_table(top_df)

col_config = {
    "Rank":            st.column_config.NumberColumn("🏅 Rank",           format="%d",  width="small"),
    "Player":          st.column_config.TextColumn("👤 Player",            width="medium"),
    "Team":            st.column_config.TextColumn("🏟 Team",              width="small"),
    "Pressure Rating": st.column_config.TextColumn("⚡ Rating",            width="small"),
    "Pressure Economy":st.column_config.TextColumn("🎯 Press Econ",        width="small"),
    "Baseline Economy":st.column_config.TextColumn("📏 Base Econ",         width="small"),
    "Economy Impr.":   st.column_config.TextColumn("📈 Impr.",             width="small"),
    "Wickets":         st.column_config.TextColumn("🎳 Wickets",           width="small"),
    "Wkts Above Exp":  st.column_config.TextColumn("🔥 Wkts+Exp",         width="small"),
    "Pressure Balls":  st.column_config.TextColumn("⚾ Balls",             width="small"),
    "Matches":         st.column_config.TextColumn("🗓 Matches",           width="small"),
    "Format Coverage": st.column_config.TextColumn("📡 Format",            width="small"),
}

st.dataframe(
    display_tbl,
    column_config=col_config,
    use_container_width=True,
    hide_index=True,
)

# ── Charts row 1 ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Performance Charts</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    fig_bar = go.Figure()
    if not top_df.empty and "adjusted_score" in top_df.columns:
        plot_df = top_df.dropna(subset=["adjusted_score"]).sort_values("adjusted_score")
        econ_vals = plot_df.get("pressure_economy", pd.Series(dtype=float)).fillna(0)

        fig_bar.add_trace(go.Bar(
            x=plot_df["adjusted_score"],
            y=plot_df["player_name"],
            orientation="h",
            marker=dict(
                color=econ_vals,
                colorscale=[[0, "#00ffc3"], [0.5, "#4A9EFF"], [1, "#1a1a6a"]],
                reversescale=True,  # lower economy = better = brighter
                showscale=True,
                colorbar=dict(title="Press Econ", tickfont=dict(color="white"), title_font=dict(color="white")),
            ),
            text=[f"{v:.2f}" for v in plot_df["adjusted_score"]],
            textposition="outside",
            textfont=dict(color="white", size=11),
            hovertemplate="<b>%{y}</b><br>Rating: %{x:.3f}<extra></extra>",
        ))
        fig_bar.update_layout(
            **_CL,
            title=dict(text="Pressure Rating (Adjusted Score)", font=dict(size=14, color="white")),
            xaxis=dict(showgrid=True, gridcolor="#1E2F4A", title="Adjusted Score"),
            yaxis=dict(showgrid=False),
            height=400,
        )
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    fig_sc = go.Figure()
    if not top_df.empty and "pressure_economy" in top_df.columns and "baseline_economy" in top_df.columns:
        sc_df = top_df.dropna(subset=["pressure_economy", "baseline_economy"])
        all_vals = pd.concat([sc_df["pressure_economy"], sc_df["baseline_economy"]])
        mn, mx = max(0, all_vals.min() * 0.9), all_vals.max() * 1.1

        fig_sc.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color="#A0ADB8", dash="dash", width=1.5),
            name="Equal (y = x)",
            showlegend=True,
        ))
        bubble_sizes = sc_df.get("pressure_balls", pd.Series([20]*len(sc_df))).fillna(20)
        bubble_sizes = (bubble_sizes - bubble_sizes.min()) / (bubble_sizes.max() - bubble_sizes.min() + 1) * 30 + 8

        fig_sc.add_trace(go.Scatter(
            x=sc_df["baseline_economy"],
            y=sc_df["pressure_economy"],
            mode="markers+text",
            text=sc_df["player_name"].apply(lambda x: x.split()[-1] if isinstance(x, str) else x),
            textposition="top center",
            textfont=dict(size=10, color="white"),
            marker=dict(
                size=bubble_sizes,
                color=_BLUE,
                opacity=0.85,
                line=dict(color="#0A0F1E", width=1),
            ),
            name="Players",
            hovertemplate="<b>%{text}</b><br>Base Econ: %{x:.2f}<br>Press Econ: %{y:.2f}<extra></extra>",
        ))
        fig_sc.update_layout(
            **_CL,
            title=dict(text="Pressure Economy vs Baseline Economy (below line = improved)", font=dict(size=14, color="white")),
            xaxis=dict(title="Baseline Economy", range=[mn, mx], showgrid=True, gridcolor="#1E2F4A"),
            yaxis=dict(title="Pressure Economy", range=[mn, mx], showgrid=True, gridcolor="#1E2F4A"),
            height=400,
        )
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    fig_runs = go.Figure()
    if not top_df.empty:
        # Expected vs Actual runs conceded
        actual_col   = "ipl_pressure_runs_conceded"  # use total if available
        # Try to build a total conceded column
        rc_candidates = [
            c for c in ["pressure_runs_conceded", "total_pressure_runs_conceded"]
            if c in top_df.columns
        ]
        # Fallback: reconstruct from economy * balls / 6
        if rc_candidates:
            r_df = top_df.dropna(subset=[rc_candidates[0]])
            names = r_df["player_name"].tolist()
            fig_runs.add_trace(go.Bar(
                x=names, y=r_df[rc_candidates[0]],
                name="Actual Runs Conceded", marker_color=_RED,
                hovertemplate="<b>%{x}</b><br>Runs Conceded: %{y:.0f}<extra></extra>",
            ))
        elif "pressure_economy" in top_df.columns and "pressure_balls" in top_df.columns:
            r_df = top_df.dropna(subset=["pressure_economy", "pressure_balls"])
            r_df = r_df.copy()
            r_df["_actual_rc"] = r_df["pressure_economy"] * r_df["pressure_balls"] / 6.0
            names = r_df["player_name"].tolist()
            fig_runs.add_trace(go.Bar(
                x=names, y=r_df["_actual_rc"],
                name="Actual Runs Conceded", marker_color=_RED,
                hovertemplate="<b>%{x}</b><br>Runs Conceded: %{y:.1f}<extra></extra>",
            ))
        else:
            r_df = top_df.copy()
            names = r_df["player_name"].tolist()

        # Expected runs conceded
        exp_col = "pressure_expected_wickets"  # not exactly runs, but closest
        # Better: use runs_prevented_pressure — it implies actual vs expected
        rp_col = "runs_prevented_pressure"
        if rp_col in top_df.columns and not r_df.empty:
            rp_df = top_df[top_df["player_name"].isin(r_df["player_name"])] if not r_df.empty else top_df
            fig_runs.add_trace(go.Bar(
                x=rp_df["player_name"],
                y=rp_df[rp_col].abs(),
                name="Runs Prevented vs Expected",
                marker_color=_GREEN,
                opacity=0.8,
                hovertemplate="<b>%{x}</b><br>Runs Prevented: %{y:.1f}<extra></extra>",
            ))
        fig_runs.update_layout(
            **_CL,
            title=dict(text="Runs Conceded & Prevented vs Expected", font=dict(size=14, color="white")),
            barmode="group",
            xaxis=dict(tickangle=-35, showgrid=False),
            yaxis=dict(title="Runs", showgrid=True, gridcolor="#1E2F4A"),
            legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
            height=400,
        )
    st.plotly_chart(fig_runs, use_container_width=True)

with c4:
    fig_dp = go.Figure()
    if not top_df.empty:
        dot_col  = "pressure_dot_pct"
        bnd_col  = "pressure_boundary_pct"

        # Try bowler_full for these columns
        dp_df = top_df.copy()
        if (dot_col not in dp_df.columns or bnd_col not in dp_df.columns) and not bowler_full.empty:
            extra = bowler_full[["player_name"] + [c for c in [dot_col, bnd_col] if c in bowler_full.columns]]
            dp_df = dp_df.merge(extra, on="player_name", how="left", suffixes=("", "_bf"))
            for c in [dot_col, bnd_col]:
                if c + "_bf" in dp_df.columns:
                    dp_df[c] = dp_df[c].combine_first(dp_df[c + "_bf"])

        has_dot = dot_col in dp_df.columns and dp_df[dot_col].notna().any()
        has_bnd = bnd_col in dp_df.columns and dp_df[bnd_col].notna().any()

        dp_df = dp_df.dropna(subset=["player_name"])

        if has_dot:
            fig_dp.add_trace(go.Bar(
                x=dp_df["player_name"],
                y=dp_df[dot_col].fillna(0) * 100 if dp_df[dot_col].max() <= 1 else dp_df[dot_col].fillna(0),
                name="Dot Ball %",
                marker_color=_GREEN,
                hovertemplate="<b>%{x}</b><br>Dot Ball %: %{y:.1f}%<extra></extra>",
            ))
        if has_bnd:
            fig_dp.add_trace(go.Bar(
                x=dp_df["player_name"],
                y=dp_df[bnd_col].fillna(0) * 100 if dp_df[bnd_col].max() <= 1 else dp_df[bnd_col].fillna(0),
                name="Boundary Conceded %",
                marker_color=_RED,
                hovertemplate="<b>%{x}</b><br>Boundary Conceded %: %{y:.1f}%<extra></extra>",
            ))
        if not has_dot and not has_bnd:
            st.info("Dot ball and boundary conceded % data not available in the current dataset.")

        fig_dp.update_layout(
            **_CL,
            title=dict(text="Dot Ball % vs Boundary Conceded %", font=dict(size=14, color="white")),
            barmode="stack",
            xaxis=dict(tickangle=-35, showgrid=False),
            yaxis=dict(title="Percentage", showgrid=True, gridcolor="#1E2F4A"),
            legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
            height=400,
        )
    st.plotly_chart(fig_dp, use_container_width=True)

# ── Radar Chart ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Player Radar — Component Breakdown</div>', unsafe_allow_html=True)

RADAR_AXES = ["Runs Prevented", "Economy Improvement", "Wicket Taking", "Dot Generation", "Boundary Prevention"]
RADAR_COLS = ["component_runs_prevented", "component_economy", "component_wickets", "component_dots", "component_boundary"]

def _z_to_score(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return 50.0
    return max(0.0, min(100.0, float(z) * 15 + 50))

radar_players = top_df["player_name"].dropna().tolist() if "player_name" in top_df.columns else []
sel_radar = st.selectbox("Select player for detailed radar chart", radar_players, key="radar_sel")

if sel_radar and not top_df.empty:
    r_row = top_df[top_df["player_name"] == sel_radar]
    if r_row.empty and not bowler_full.empty and "player_name" in bowler_full.columns:
        r_row = bowler_full[bowler_full["player_name"] == sel_radar]

    if not r_row.empty:
        r_row = r_row.iloc[0]
        radar_vals = [_z_to_score(r_row.get(c, np.nan)) for c in RADAR_COLS]
        radar_vals_closed = radar_vals + [radar_vals[0]]
        axes_closed       = RADAR_AXES + [RADAR_AXES[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals_closed,
            theta=axes_closed,
            fill="toself",
            fillcolor="rgba(74,158,255,0.2)",
            line=dict(color=_BLUE, width=2),
            name=sel_radar,
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[50] * (len(RADAR_AXES) + 1),
            theta=axes_closed,
            mode="lines",
            line=dict(color="#A0ADB8", width=1, dash="dash"),
            name="Average (50)",
            showlegend=True,
        ))
        fig_radar.update_layout(
            paper_bgcolor="#1E2538",
            plot_bgcolor="#1E2538",
            font=dict(color="white", size=12),
            polar=dict(
                bgcolor="#1E2538",
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor="#1E2F4A", tickfont=dict(color="#A0ADB8", size=10),
                    tickvals=[0, 25, 50, 75, 100],
                ),
                angularaxis=dict(gridcolor="#1E2F4A", tickfont=dict(color="white", size=12)),
            ),
            title=dict(text=f"Component Radar: {sel_radar}", font=dict(size=16, color="white"), x=0.5),
            showlegend=True,
            legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
            height=500,
            margin=dict(l=60, r=60, t=80, b=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ── Player Profile ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">👤 Player Profile</div>', unsafe_allow_html=True)

profile_players = top_df["player_name"].dropna().tolist() if "player_name" in top_df.columns else []
sel_player = st.selectbox("Select player for full profile", profile_players, key="profile_sel")

if sel_player and not top_df.empty:
    p = top_df[top_df["player_name"] == sel_player]
    if not p.empty:
        p = p.iloc[0]

        pf = pd.Series(dtype=object)
        if not bowler_full.empty and "player_name" in bowler_full.columns:
            pf_rows = bowler_full[bowler_full["player_name"] == sel_player]
            if not pf_rows.empty:
                pf = pf_rows.iloc[0]

        def _g(key):
            v = pf.get(key, None) if not pf.empty else None
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = p.get(key, None)
            return v

        rank_val   = _v(_g("rank"), "int")
        team_val   = str(_g("main_team_name") or "Unknown")
        format_val = str(_g("format_coverage") or "Unknown")

        with st.expander(f"📋 Full Profile: {sel_player}", expanded=True):
            # Header
            st.markdown(
                f"""
                <div class="profile-header">
                  <div class="profile-name">{sel_player}</div>
                  <div class="profile-team">🏟 {team_val} &nbsp;|&nbsp; 📡 {format_val}</div>
                  <div class="rank-pill">🏅 Rank #{rank_val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Stat row 1
            s1, s2, s3, s4 = st.columns(4)
            for col_w, lbl, key, fmt in [
                (s1, "Pressure Economy",   "pressure_economy",   "2f"),
                (s2, "Baseline Economy",   "baseline_economy",   "2f"),
                (s3, "Economy Impr.",       "economy_improvement","+1f"),
                (s4, "Pressure Balls",     "pressure_balls",     "int"),
            ]:
                with col_w:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value">{_v(_g(key), fmt)}</div>'
                        f'<div class="stat-label">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # Stat row 2
            s5, s6, s7, s8 = st.columns(4)
            for col_w, lbl, key, fmt in [
                (s5, "Pressure Wickets",     "pressure_wickets",          "int"),
                (s6, "Expected Wickets",     "pressure_expected_wickets", "1f"),
                (s7, "Wickets Above Exp",    "wickets_above_expected",    "+1f"),
                (s8, "Matches",              "pressure_matches",          "int"),
            ]:
                with col_w:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value">{_v(_g(key), fmt)}</div>'
                        f'<div class="stat-label">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # IPL vs T20I
            st.markdown("**IPL vs T20I Breakdown**")
            ipl_col_w, t20_col_w = st.columns(2)

            with ipl_col_w:
                st.markdown(
                    f"""
                    <div class="format-mini">
                      <div class="format-title">🟦 IPL</div>
                      <div class="format-val">{_v(_g('ipl_pressure_economy'), '2f')}</div>
                      <div class="format-sub">Pressure Economy</div>
                      <div style="margin-top:8px;font-size:0.82rem;color:#A0ADB8;">
                        RC: {_v(_g('ipl_pressure_runs_conceded'), 'int')} &nbsp;|&nbsp;
                        Balls: {_v(_g('ipl_pressure_balls'), 'int')} &nbsp;|&nbsp;
                        Wkts: {_v(_g('ipl_pressure_wickets'), 'int')} &nbsp;|&nbsp;
                        Matches: {_v(_g('ipl_pressure_matches'), 'int')}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with t20_col_w:
                st.markdown(
                    f"""
                    <div class="format-mini">
                      <div class="format-title">🟩 T20I</div>
                      <div class="format-val">{_v(_g('t20i_pressure_economy'), '2f')}</div>
                      <div class="format-sub">Pressure Economy</div>
                      <div style="margin-top:8px;font-size:0.82rem;color:#A0ADB8;">
                        RC: {_v(_g('t20i_pressure_runs_conceded'), 'int')} &nbsp;|&nbsp;
                        Balls: {_v(_g('t20i_pressure_balls'), 'int')} &nbsp;|&nbsp;
                        Wkts: {_v(_g('t20i_pressure_wickets'), 'int')} &nbsp;|&nbsp;
                        Matches: {_v(_g('t20i_pressure_matches'), 'int')}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Statistical justification
            def _justify_bowler(row_p, row_pf):
                name   = str(row_p.get("player_name", sel_player))
                rank_n = _v(row_p.get("rank"),                  "int")
                bopi   = _v(row_pf.get("avg_bopi") if not row_pf.empty else None, "1f")
                pecon  = _v(row_p.get("pressure_economy"),      "2f")
                becon  = _v(row_p.get("baseline_economy"),      "2f")
                imp    = _v(row_p.get("economy_improvement"),   "+1f")
                pwkts  = _v(row_p.get("pressure_wickets"),      "int")
                wae    = _v(row_p.get("wickets_above_expected"), "+1f")
                rp     = _v(row_p.get("runs_prevented_pressure"),"+1f")
                dots   = _v(row_pf.get("pressure_dot_pct") if not row_pf.empty else None, "pct")
                bnd    = _v(row_pf.get("pressure_boundary_pct") if not row_pf.empty else None, "pct")
                pballs = _v(row_p.get("pressure_balls"), "int")
                pmatch = _v(row_p.get("pressure_matches"), "int")

                parts = [f"**Why {name} ranks #{rank_n}:**\n\n"]
                bopi_txt = f" (avg BoPI: {bopi})" if bopi != "N/A" else ""
                parts.append(
                    f"Under high-pressure situations{bopi_txt}, {name} concedes at an economy "
                    f"of **{pecon}** compared to their baseline of **{becon}** — "
                    f"an improvement of **{imp}** runs per over. "
                )
                if wae != "N/A":
                    parts.append(f"They take **{wae}** wickets above what the model expects under pressure. ")
                if rp != "N/A":
                    parts.append(f"They prevent **{rp}** runs above expected, tightening the game. ")
                if dots != "N/A":
                    parts.append(f"Their dot ball percentage under pressure is **{dots}**, ")
                if bnd != "N/A":
                    parts.append(f"conceding just **{bnd}** of deliveries to the boundary. ")
                parts.append(
                    f"This rating is based on **{pwkts}** pressure wickets across **{pballs}** "
                    f"deliveries in **{pmatch}** matches."
                )
                return "".join(parts)

            justify_text = _justify_bowler(p, pf)
            st.markdown(
                f'<div class="justify-box">{justify_text}</div>',
                unsafe_allow_html=True,
            )

            ci_lo = _v(_g("ci_lower"), "2f")
            ci_hi = _v(_g("ci_upper"), "2f")
            pballs_n = _v(_g("pressure_balls"), "int")
            pmatch_n = _v(_g("pressure_matches"), "int")

            st.markdown(
                f"""
                <div class="ci-box">
                  📐 <strong>Pressure Economy confidence interval:</strong> {ci_lo} – {ci_hi}
                  &nbsp;&nbsp;|&nbsp;&nbsp;
                  🗂 <strong>Sample:</strong> {pballs_n} pressure deliveries across {pmatch_n} matches
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Wicket Chart ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎳 Wicket Analysis</div>', unsafe_allow_html=True)

if not top_df.empty and "pressure_wickets" in top_df.columns:
    wk_df = top_df.dropna(subset=["pressure_wickets"])

    fig_wk = go.Figure()
    fig_wk.add_trace(go.Bar(
        x=wk_df["player_name"],
        y=wk_df["pressure_wickets"],
        name="Pressure Wickets",
        marker_color=_BLUE,
        hovertemplate="<b>%{x}</b><br>Wickets: %{y}<extra></extra>",
    ))
    if "wickets_above_expected" in wk_df.columns and wk_df["wickets_above_expected"].notna().any():
        fig_wk.add_trace(go.Bar(
            x=wk_df["player_name"],
            y=wk_df["wickets_above_expected"],
            name="Wickets Above Expected",
            marker_color=_GREEN,
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Wkts Above Exp: %{y:+.1f}<extra></extra>",
        ))
    fig_wk.update_layout(
        **_CL,
        title=dict(text="Pressure Wickets & Wickets Above Expected", font=dict(size=14, color="white")),
        barmode="group",
        xaxis=dict(tickangle=-35, showgrid=False),
        yaxis=dict(title="Wickets", showgrid=True, gridcolor="#1E2F4A"),
        legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
        height=380,
    )
    st.plotly_chart(fig_wk, use_container_width=True)
else:
    st.info("Wicket data not available in the current filtered dataset.")
