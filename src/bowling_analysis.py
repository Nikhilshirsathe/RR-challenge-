"""
bowling_analysis.py – Pressure-bowling metrics for the SUPRR Selector.

Exports
-------
ELIGIBILITY_THRESHOLDS : dict
    Default eligibility cut-offs (total_balls, pressure_balls, pressure_matches).

compute_bowler_pressure_stats(df, bowler_baselines_df=None) -> pd.DataFrame
    One row per bowler_id containing every pressure-bowling metric,
    baseline-comparison columns, and an eligibility flag.

get_eligible_bowlers(df, bowler_baselines_df=None, thresholds=None) -> pd.DataFrame
    Thin wrapper that filters compute_bowler_pressure_stats to eligible players.

get_bowler_analysis_summary(stats_df) -> dict
    High-level summary stats over the computed/eligible result frame.

Pressure is defined by the ``is_high_pressure_bowl`` boolean column produced by
pressure_engine.py.  Expected-performance columns (expected_runs_conceded_per_ball,
expected_wicket_prob) are consumed when present but never required – missing
columns yield NaN gracefully.

All heavy computation is fully vectorised (pandas groupby / numpy) with no
Python-level row loops.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streamlit cache shim – degrades gracefully without Streamlit
# ---------------------------------------------------------------------------
try:
    import streamlit as st  # type: ignore[import]
    _cache = st.cache_data
    logger.debug("bowling_analysis: Streamlit detected – @st.cache_data available.")
except ImportError:
    def _cache(func=None, **_kwargs: Any):  # type: ignore[misc]
        if func is not None:
            return func
        return lambda f: f
    logger.debug("bowling_analysis: Streamlit not available – using no-op cache decorator.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELIGIBILITY_THRESHOLDS: dict[str, int] = {
    "min_total_balls": 120,
    "min_pressure_balls": 24,
    "min_pressure_matches": 5,
}

# Dismissal types that do NOT credit the bowler (kept in sync with player_baseline.py)
_NON_BOWLER_DISMISSALS: frozenset[str] = frozenset({
    "RunOut",
    "Run Out",
    "run out",
    "Hit Wicket",
    "hit wicket",
    "Obstracting the field",   # intentional typo from spec
    "Obstructing the field",
    "Obstructing The Field",
    "Handled the ball",
    "Retired hurt",
    "Retired out",
})

# Minimum pressure balls to report a meaningful economy figure
_MIN_ECONOMY_BALLS = 6

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _most_common(series: pd.Series) -> Any:
    """Return the most frequent non-null value, or pd.NA if series is empty."""
    counts = series.dropna().value_counts()
    return counts.index[0] if len(counts) else pd.NA


def _is_bowler_credit_col(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series: True when the wicket is credited to the bowler.

    Excludes run-outs, hit wickets, obstructions, and other non-bowler modes.
    If 'dismissal_type' is absent, all is_wicket rows are credited (with a
    warning logged).
    """
    if "dismissal_type" not in df.columns:
        logger.warning(
            "bowling_analysis._is_bowler_credit_col: 'dismissal_type' column "
            "not found – crediting all is_wicket rows to the bowler."
        )
        return df["is_wicket"].astype(bool)

    dismissal = df["dismissal_type"].fillna("").astype(str).str.strip()
    return df["is_wicket"].astype(bool) & ~dismissal.isin(_NON_BOWLER_DISMISSALS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@_cache
def compute_bowler_pressure_stats(
    df: pd.DataFrame,
    bowler_baselines_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-bowler pressure metrics from an enriched ball-by-ball frame.

    Parameters
    ----------
    df : pd.DataFrame
        Full enriched + pressure-annotated ball-by-ball dataframe.
    bowler_baselines_df : pd.DataFrame, optional
        Output of ``player_baseline.compute_bowler_baselines``.  When supplied,
        baseline columns (overall_economy, dot_ball_pct, boundary_conceded_pct)
        are merged in and comparison deltas are computed.

    Returns
    -------
    pd.DataFrame
        One row per bowler_id.  Index = bowler_id.

    Raises
    ------
    ValueError
        If mandatory columns are absent from *df*.
    """
    if df is None or df.empty:
        logger.warning("compute_bowler_pressure_stats: received empty dataframe.")
        return pd.DataFrame()

    logger.info("compute_bowler_pressure_stats: processing %d rows.", len(df))

    # ------------------------------------------------------------------ #
    # Validate required columns
    # ------------------------------------------------------------------ #
    required = {
        "bowler_id", "bowler_name", "bowling_team_id", "match_id",
        "is_legal_ball", "runs_charged_to_bowler",
        "is_boundary_4", "is_boundary_6", "is_wicket",
        "is_high_pressure_bowl", "bopi",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_bowler_pressure_stats: missing required columns: {missing}"
        )

    # ------------------------------------------------------------------ #
    # Work exclusively on legal deliveries
    # ------------------------------------------------------------------ #
    legal = df[df["is_legal_ball"].astype(bool)].copy()
    logger.info(
        "compute_bowler_pressure_stats: %d legal deliveries from %d total.",
        len(legal), len(df),
    )

    # Bowler-credit wicket flag on the full legal frame
    legal["_bowler_wicket"] = _is_bowler_credit_col(legal)

    # ------------------------------------------------------------------ #
    # Pressure subset
    # ------------------------------------------------------------------ #
    pressure = legal[legal["is_high_pressure_bowl"].astype(bool)].copy()
    logger.info(
        "compute_bowler_pressure_stats: %d high-pressure legal deliveries.",
        len(pressure),
    )

    # Pre-compute helper columns inside the pressure frame
    pressure["_is_boundary"] = (
        pressure["is_boundary_4"].astype(bool) | pressure["is_boundary_6"].astype(bool)
    )
    pressure["_is_dot_bowl"] = pressure["runs_charged_to_bowler"] == 0

    # ------------------------------------------------------------------ #
    # Core pressure aggregation (single groupby pass)
    # ------------------------------------------------------------------ #
    pgrp = pressure.groupby("bowler_id", sort=True)

    pressure_balls    = pgrp["is_legal_ball"].count().rename("pressure_balls")
    pressure_runs_con = pgrp["runs_charged_to_bowler"].sum().rename("pressure_runs_conceded")
    pressure_wickets  = pgrp["_bowler_wicket"].sum().astype(int).rename("pressure_wickets")
    pressure_dots     = pgrp["_is_dot_bowl"].sum().rename("pressure_dots")
    pressure_bnd      = pgrp["_is_boundary"].sum().rename("pressure_boundaries_conceded")
    pressure_matches  = pgrp["match_id"].nunique().rename("pressure_matches")
    avg_bopi          = pgrp["bopi"].mean().rename("avg_bopi")

    # Expected columns (optional)
    if "expected_runs_conceded_per_ball" in pressure.columns:
        exp_runs = pgrp["expected_runs_conceded_per_ball"].sum().rename("pressure_expected_runs")
        logger.debug("compute_bowler_pressure_stats: expected_runs_conceded_per_ball present.")
    else:
        exp_runs = pd.Series(np.nan, index=pressure_balls.index, name="pressure_expected_runs")
        logger.info(
            "compute_bowler_pressure_stats: expected_runs_conceded_per_ball not found – "
            "pressure_expected_runs will be NaN."
        )

    if "expected_wicket_prob" in pressure.columns:
        exp_wkts = pgrp["expected_wicket_prob"].sum().rename("pressure_expected_wickets")
        logger.debug("compute_bowler_pressure_stats: expected_wicket_prob present.")
    else:
        exp_wkts = pd.Series(np.nan, index=pressure_balls.index, name="pressure_expected_wickets")
        logger.info(
            "compute_bowler_pressure_stats: expected_wicket_prob not found – "
            "pressure_expected_wickets will be NaN."
        )

    # Player info
    player_name = pgrp["bowler_name"].agg(_most_common).rename("player_name")
    main_team   = pgrp["bowling_team_id"].agg(_most_common).rename("main_team_id")

    # ------------------------------------------------------------------ #
    # Assemble pressure stats frame
    # ------------------------------------------------------------------ #
    stats = pd.concat(
        [
            player_name, main_team,
            pressure_balls, pressure_runs_con,
            pressure_wickets, pressure_dots, pressure_bnd,
            pressure_matches, avg_bopi,
            exp_runs, exp_wkts,
        ],
        axis=1,
    )

    # ------------------------------------------------------------------ #
    # Derived rates
    # ------------------------------------------------------------------ #
    # Overs
    stats["pressure_overs"] = stats["pressure_balls"] / 6.0

    # Economy: NaN when fewer than _MIN_ECONOMY_BALLS
    stats["pressure_economy"] = np.where(
        stats["pressure_balls"] >= _MIN_ECONOMY_BALLS,
        stats["pressure_runs_conceded"] / np.maximum(stats["pressure_overs"], 1e-9),
        np.nan,
    )

    stats["pressure_dot_pct"] = np.where(
        stats["pressure_balls"] > 0,
        stats["pressure_dots"] / stats["pressure_balls"] * 100.0,
        np.nan,
    )
    stats["pressure_boundary_pct"] = np.where(
        stats["pressure_balls"] > 0,
        stats["pressure_boundaries_conceded"] / stats["pressure_balls"] * 100.0,
        np.nan,
    )

    # Runs prevented and wickets above expected
    stats["runs_prevented_pressure"] = (
        stats["pressure_expected_runs"] - stats["pressure_runs_conceded"]
    )
    stats["wickets_above_expected"] = (
        stats["pressure_wickets"] - stats["pressure_expected_wickets"]
    )

    logger.info(
        "compute_bowler_pressure_stats: core pressure stats done for %d bowlers.",
        len(stats),
    )

    # ------------------------------------------------------------------ #
    # IPL vs T20I pressure splits
    # ------------------------------------------------------------------ #
    def _pressure_format_split(fmt_label: str, prefix: str) -> pd.DataFrame:
        """Compute pressure bowling stats for a single match format."""
        col_names = [
            f"{prefix}_pressure_balls",
            f"{prefix}_pressure_runs_conceded",
            f"{prefix}_pressure_economy",
            f"{prefix}_pressure_wickets",
            f"{prefix}_pressure_matches",
        ]
        if "match_format" not in pressure.columns:
            logger.warning(
                "compute_bowler_pressure_stats: 'match_format' column missing – "
                "%s pressure split will be NaN.", fmt_label
            )
            return pd.DataFrame(columns=col_names, index=stats.index, dtype=float)

        sub = pressure[pressure["match_format"].astype(str).str.upper() == fmt_label.upper()]
        if sub.empty:
            return pd.DataFrame(columns=col_names, index=stats.index, dtype=float)

        sg = sub.groupby("bowler_id", sort=True)
        balls   = sg["is_legal_ball"].count().rename(f"{prefix}_pressure_balls")
        runs    = sg["runs_charged_to_bowler"].sum().rename(f"{prefix}_pressure_runs_conceded")
        wkts    = sg["_bowler_wicket"].sum().astype(int).rename(f"{prefix}_pressure_wickets")
        matches = sg["match_id"].nunique().rename(f"{prefix}_pressure_matches")
        split   = pd.concat([balls, runs, wkts, matches], axis=1)
        split[f"{prefix}_pressure_economy"] = np.where(
            split[f"{prefix}_pressure_balls"] >= _MIN_ECONOMY_BALLS,
            split[f"{prefix}_pressure_runs_conceded"] / (
                split[f"{prefix}_pressure_balls"] / 6.0
            ),
            np.nan,
        )
        return split

    ipl_split  = _pressure_format_split("T20",  "ipl")
    t20i_split = _pressure_format_split("T20I", "t20i")

    stats = stats.join(ipl_split,  how="left")
    stats = stats.join(t20i_split, how="left")

    # Fill integer count columns with 0 where NaN
    split_count_cols = [
        c for c in stats.columns
        if c in {
            "ipl_pressure_balls", "ipl_pressure_runs_conceded",
            "ipl_pressure_wickets", "ipl_pressure_matches",
            "t20i_pressure_balls", "t20i_pressure_runs_conceded",
            "t20i_pressure_wickets", "t20i_pressure_matches",
        }
    ]
    stats[split_count_cols] = stats[split_count_cols].fillna(0).astype(int)

    # ------------------------------------------------------------------ #
    # Overall total_balls (from baselines if available; else recompute)
    # ------------------------------------------------------------------ #
    if bowler_baselines_df is not None and not bowler_baselines_df.empty:
        if "total_legal_balls" in bowler_baselines_df.columns:
            stats["total_balls"] = (
                bowler_baselines_df["total_legal_balls"].reindex(stats.index)
            )
        else:
            logger.warning(
                "compute_bowler_pressure_stats: bowler_baselines_df has no "
                "'total_legal_balls' column – recomputing total_balls."
            )
            stats["total_balls"] = (
                legal.groupby("bowler_id")["is_legal_ball"].count().reindex(stats.index)
            )
    else:
        stats["total_balls"] = (
            legal.groupby("bowler_id")["is_legal_ball"].count().reindex(stats.index)
        )

    # ------------------------------------------------------------------ #
    # Baseline comparison (requires bowler_baselines_df)
    # ------------------------------------------------------------------ #
    if bowler_baselines_df is not None and not bowler_baselines_df.empty:
        logger.info("compute_bowler_pressure_stats: merging bowler baselines.")
        bl = bowler_baselines_df.reindex(stats.index)

        stats["baseline_economy"]   = bl.get("overall_economy",         pd.Series(dtype=float))
        stats["normal_dot_pct"]     = bl.get("dot_ball_pct",            pd.Series(dtype=float))
        stats["normal_boundary_pct"] = bl.get("boundary_conceded_pct", pd.Series(dtype=float))

        # Positive = bowler is better (cheaper) under pressure
        stats["economy_improvement"] = stats["baseline_economy"] - stats["pressure_economy"]
        # Positive = more dots under pressure than normal
        stats["dot_improvement"] = stats["pressure_dot_pct"] - stats["normal_dot_pct"]
        # Positive = fewer boundaries conceded under pressure
        stats["boundary_prevention_improvement"] = (
            stats["normal_boundary_pct"] - stats["pressure_boundary_pct"]
        )

        # Fill player_name from baselines for IDs absent from pressure subset
        if "player_name" in bl.columns:
            stats["player_name"] = stats["player_name"].where(
                stats["player_name"].notna(), bl["player_name"]
            )

        logger.info("compute_bowler_pressure_stats: baseline comparison columns added.")
    else:
        logger.info(
            "compute_bowler_pressure_stats: no bowler_baselines_df supplied – "
            "baseline comparison columns will be NaN."
        )
        stats["baseline_economy"]               = np.nan
        stats["normal_dot_pct"]                 = np.nan
        stats["normal_boundary_pct"]            = np.nan
        stats["economy_improvement"]            = np.nan
        stats["dot_improvement"]                = np.nan
        stats["boundary_prevention_improvement"] = np.nan

    # ------------------------------------------------------------------ #
    # Eligibility flag
    # ------------------------------------------------------------------ #
    stats["is_eligible"] = (
        (stats["total_balls"].fillna(0)        >= ELIGIBILITY_THRESHOLDS["min_total_balls"])
        & (stats["pressure_balls"].fillna(0)   >= ELIGIBILITY_THRESHOLDS["min_pressure_balls"])
        & (stats["pressure_matches"].fillna(0) >= ELIGIBILITY_THRESHOLDS["min_pressure_matches"])
    )

    eligible_count = int(stats["is_eligible"].sum())
    logger.info(
        "compute_bowler_pressure_stats: %d / %d bowlers are eligible.",
        eligible_count, len(stats),
    )

    # ------------------------------------------------------------------ #
    # Final column ordering
    # ------------------------------------------------------------------ #
    ordered_cols = [
        # Identity
        "player_name", "main_team_id",
        # Overall
        "total_balls",
        # Pressure core
        "pressure_balls", "pressure_overs", "pressure_runs_conceded", "pressure_economy",
        "pressure_wickets",
        "pressure_dots", "pressure_dot_pct",
        "pressure_boundaries_conceded", "pressure_boundary_pct",
        "pressure_matches", "avg_bopi",
        # Expected performance
        "pressure_expected_runs", "pressure_expected_wickets",
        "runs_prevented_pressure", "wickets_above_expected",
        # Format splits
        "ipl_pressure_balls", "ipl_pressure_runs_conceded", "ipl_pressure_economy",
        "ipl_pressure_wickets", "ipl_pressure_matches",
        "t20i_pressure_balls", "t20i_pressure_runs_conceded", "t20i_pressure_economy",
        "t20i_pressure_wickets", "t20i_pressure_matches",
        # Baseline comparison
        "baseline_economy", "economy_improvement",
        "normal_dot_pct", "dot_improvement",
        "normal_boundary_pct", "boundary_prevention_improvement",
        # Eligibility
        "is_eligible",
    ]
    ordered_cols = [c for c in ordered_cols if c in stats.columns]
    extra_cols = [c for c in stats.columns if c not in ordered_cols]
    result = stats[ordered_cols + extra_cols].copy()
    result.index.name = "bowler_id"

    logger.info(
        "compute_bowler_pressure_stats: complete – %d bowlers, %d eligible.",
        len(result), eligible_count,
    )
    return result


def get_eligible_bowlers(
    df: pd.DataFrame,
    bowler_baselines_df: pd.DataFrame | None = None,
    thresholds: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Return only bowlers who meet the eligibility thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Full enriched ball-by-ball dataframe.
    bowler_baselines_df : pd.DataFrame, optional
        Bowler baseline frame from ``player_baseline.compute_bowler_baselines``.
    thresholds : dict, optional
        Override dict with keys ``min_total_balls``, ``min_pressure_balls``,
        ``min_pressure_matches``.  Defaults to ``ELIGIBILITY_THRESHOLDS``.

    Returns
    -------
    pd.DataFrame
        Filtered stats frame containing only eligible bowlers.
    """
    effective = {**ELIGIBILITY_THRESHOLDS, **(thresholds or {})}
    logger.info(
        "get_eligible_bowlers: thresholds = total_balls>=%d, "
        "pressure_balls>=%d, pressure_matches>=%d.",
        effective["min_total_balls"],
        effective["min_pressure_balls"],
        effective["min_pressure_matches"],
    )

    stats = compute_bowler_pressure_stats(df, bowler_baselines_df)
    if stats.empty:
        logger.warning("get_eligible_bowlers: stats frame is empty – returning empty.")
        return stats

    mask = (
        (stats["total_balls"].fillna(0)        >= effective["min_total_balls"])
        & (stats["pressure_balls"].fillna(0)   >= effective["min_pressure_balls"])
        & (stats["pressure_matches"].fillna(0) >= effective["min_pressure_matches"])
    )
    eligible = stats[mask].copy()

    logger.info(
        "get_eligible_bowlers: %d / %d bowlers pass eligibility.",
        len(eligible), len(stats),
    )
    return eligible


def get_bowler_analysis_summary(stats_df: pd.DataFrame) -> dict[str, Any]:
    """Return a high-level summary dict from a bowler stats / eligible frame.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output of ``compute_bowler_pressure_stats`` or ``get_eligible_bowlers``.

    Returns
    -------
    dict with keys:
        n_eligible              – number of eligible rows in *stats_df*
        avg_pressure_economy    – mean pressure economy across eligible players
        avg_pressure_balls      – mean pressure balls bowled across eligible players
        top_player_name         – player with best (lowest) pressure_economy
        top_player_economy      – that player's pressure_economy
    """
    if stats_df is None or stats_df.empty:
        logger.warning("get_bowler_analysis_summary: empty stats_df received.")
        return {
            "n_eligible": 0,
            "avg_pressure_economy": np.nan,
            "avg_pressure_balls": np.nan,
            "top_player_name": None,
            "top_player_economy": np.nan,
        }

    # Restrict to eligible rows if the flag is present
    if "is_eligible" in stats_df.columns:
        eligible = stats_df[stats_df["is_eligible"].astype(bool)]
    else:
        eligible = stats_df

    n_eligible          = int(len(eligible))
    avg_pressure_econ   = float(eligible["pressure_economy"].mean())  if "pressure_economy" in eligible.columns else np.nan
    avg_pressure_balls  = float(eligible["pressure_balls"].mean())    if "pressure_balls"   in eligible.columns else np.nan

    top_player_name: str | None = None
    top_player_economy: float = np.nan
    if "pressure_economy" in eligible.columns and not eligible.empty:
        # Best economy = lowest value (dropna so NaN rows are ignored)
        econ_series = eligible["pressure_economy"].dropna()
        if not econ_series.empty:
            top_idx = econ_series.idxmin()
            top_player_economy = float(econ_series.loc[top_idx])
            top_player_name = str(
                eligible.loc[top_idx, "player_name"]
                if "player_name" in eligible.columns
                else top_idx
            )

    summary = {
        "n_eligible": n_eligible,
        "avg_pressure_economy": round(avg_pressure_econ, 2) if not np.isnan(avg_pressure_econ) else np.nan,
        "avg_pressure_balls": round(avg_pressure_balls, 1) if not np.isnan(avg_pressure_balls) else np.nan,
        "top_player_name": top_player_name,
        "top_player_economy": round(top_player_economy, 2) if not np.isnan(top_player_economy) else np.nan,
    }
    logger.info(
        "get_bowler_analysis_summary: n_eligible=%d, avg_pressure_economy=%.2f, "
        "top=%s (economy %.2f).",
        summary["n_eligible"],
        summary["avg_pressure_economy"] if not np.isnan(summary["avg_pressure_economy"]) else -1,
        summary["top_player_name"],
        summary["top_player_economy"] if not np.isnan(summary["top_player_economy"]) else -1,
    )
    return summary
