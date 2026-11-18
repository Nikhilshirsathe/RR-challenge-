"""
batting_analysis.py – Pressure-batting metrics for the SUPRR Selector.

Exports
-------
ELIGIBILITY_THRESHOLDS : dict
    Default eligibility cut-offs (total_balls, pressure_balls, pressure_matches).

compute_batter_pressure_stats(df, batter_baselines_df=None) -> pd.DataFrame
    One row per striker_id containing every pressure-batting metric,
    baseline-comparison columns, and an eligibility flag.

get_eligible_batters(df, batter_baselines_df=None, thresholds=None) -> pd.DataFrame
    Thin wrapper that filters compute_batter_pressure_stats to eligible players.

get_batter_analysis_summary(stats_df) -> dict
    High-level summary stats over the computed/eligible result frame.

Pressure is defined by the ``is_high_pressure_bat`` boolean column produced
by pressure_engine.py.  Expected-performance columns (expected_runs_per_ball,
runs_above_expected) are consumed when present but never required – missing
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
    logger.debug("batting_analysis: Streamlit detected – @st.cache_data available.")
except ImportError:
    def _cache(func=None, **_kwargs: Any):  # type: ignore[misc]
        if func is not None:
            return func
        return lambda f: f
    logger.debug("batting_analysis: Streamlit not available – using no-op cache decorator.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELIGIBILITY_THRESHOLDS: dict[str, int] = {
    "min_total_balls": 150,
    "min_pressure_balls": 30,
    "min_pressure_matches": 5,
}

# Minimum balls in a pressure split to report a split strike rate
_MIN_SPLIT_BALLS = 1

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _most_common(series: pd.Series) -> Any:
    """Return the most frequent non-null value, or pd.NA if series is empty."""
    counts = series.dropna().value_counts()
    return counts.index[0] if len(counts) else pd.NA


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if it exists, else a Series of NaN with same index."""
    if col in df.columns:
        return df[col]
    logger.debug("batting_analysis: column '%s' not found – using NaN.", col)
    return pd.Series(np.nan, index=df.index, name=col)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@_cache
def compute_batter_pressure_stats(
    df: pd.DataFrame,
    batter_baselines_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-batter pressure metrics from an enriched ball-by-ball frame.

    Parameters
    ----------
    df : pd.DataFrame
        Full enriched + pressure-annotated ball-by-ball dataframe.
    batter_baselines_df : pd.DataFrame, optional
        Output of ``player_baseline.compute_batter_baselines``.  When supplied,
        baseline columns (overall_strike_rate, normal_boundary_pct, etc.) are
        merged in and comparison deltas are computed.

    Returns
    -------
    pd.DataFrame
        One row per striker_id.  Index = striker_id.

    Raises
    ------
    ValueError
        If mandatory columns are absent from *df*.
    """
    if df is None or df.empty:
        logger.warning("compute_batter_pressure_stats: received empty dataframe.")
        return pd.DataFrame()

    logger.info("compute_batter_pressure_stats: processing %d rows.", len(df))

    # ------------------------------------------------------------------ #
    # Validate required columns
    # ------------------------------------------------------------------ #
    required = {
        "striker_id", "striker_name", "batting_team_id", "match_id",
        "is_legal_ball", "runs_off_bat",
        "is_boundary_4", "is_boundary_6", "is_wicket",
        "is_high_pressure_bat", "bpi",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_batter_pressure_stats: missing required columns: {missing}"
        )

    # ------------------------------------------------------------------ #
    # Work exclusively on legal deliveries
    # ------------------------------------------------------------------ #
    legal = df[df["is_legal_ball"].astype(bool)].copy()
    logger.info(
        "compute_batter_pressure_stats: %d legal deliveries from %d total.",
        len(legal), len(df),
    )

    # ------------------------------------------------------------------ #
    # Pressure subset
    # ------------------------------------------------------------------ #
    pressure = legal[legal["is_high_pressure_bat"].astype(bool)].copy()
    logger.info(
        "compute_batter_pressure_stats: %d high-pressure legal deliveries.",
        len(pressure),
    )

    # Pre-compute helper columns inside the pressure frame
    pressure["_is_boundary"] = (
        pressure["is_boundary_4"].astype(bool) | pressure["is_boundary_6"].astype(bool)
    )
    pressure["_is_dot"] = (pressure["runs_off_bat"] == 0) & (~pressure["is_wicket"].astype(bool))

    # ------------------------------------------------------------------ #
    # Core pressure aggregation (one pass)
    # ------------------------------------------------------------------ #
    pgrp = pressure.groupby("striker_id", sort=True)

    pressure_balls    = pgrp["is_legal_ball"].count().rename("pressure_balls")
    pressure_runs     = pgrp["runs_off_bat"].sum().rename("pressure_runs")
    pressure_bnd      = pgrp["_is_boundary"].sum().rename("pressure_boundaries")
    pressure_dots     = pgrp["_is_dot"].sum().rename("pressure_dots")
    pressure_wkts     = pgrp["is_wicket"].sum().rename("pressure_wickets")
    pressure_matches  = pgrp["match_id"].nunique().rename("pressure_matches")
    avg_bpi           = pgrp["bpi"].mean().rename("avg_bpi")

    # Expected runs (optional column)
    if "expected_runs_per_ball" in pressure.columns:
        exp_runs = pgrp["expected_runs_per_ball"].sum().rename("expected_pressure_runs")
        logger.debug("compute_batter_pressure_stats: expected_runs_per_ball present.")
    else:
        exp_runs = pd.Series(np.nan, index=pressure_balls.index, name="expected_pressure_runs")
        logger.info(
            "compute_batter_pressure_stats: expected_runs_per_ball not found – "
            "expected_pressure_runs will be NaN."
        )

    # Player info from pressure frame (may be absent for some IDs in edge cases)
    player_name = pgrp["striker_name"].agg(_most_common).rename("player_name")
    main_team   = pgrp["batting_team_id"].agg(_most_common).rename("main_team_id")

    # ------------------------------------------------------------------ #
    # Assemble pressure stats frame
    # ------------------------------------------------------------------ #
    stats = pd.concat(
        [
            player_name, main_team,
            pressure_balls, pressure_runs,
            pressure_bnd, pressure_dots, pressure_wkts,
            pressure_matches, avg_bpi, exp_runs,
        ],
        axis=1,
    )

    # Derived rates
    stats["pressure_strike_rate"] = np.where(
        stats["pressure_balls"] > 0,
        stats["pressure_runs"] / stats["pressure_balls"] * 100.0,
        np.nan,
    )
    stats["pressure_boundary_pct"] = np.where(
        stats["pressure_balls"] > 0,
        stats["pressure_boundaries"] / stats["pressure_balls"] * 100.0,
        np.nan,
    )
    stats["pressure_dot_pct"] = np.where(
        stats["pressure_balls"] > 0,
        stats["pressure_dots"] / stats["pressure_balls"] * 100.0,
        np.nan,
    )
    stats["pressure_dismissal_rate"] = np.where(
        stats["pressure_balls"] > 0,
        stats["pressure_wickets"] / stats["pressure_balls"] * 100.0,
        np.nan,
    )
    stats["runs_above_expected_pressure"] = (
        stats["pressure_runs"] - stats["expected_pressure_runs"]
    )
    stats["pressure_runs_per_match"] = np.where(
        stats["pressure_matches"] > 0,
        stats["pressure_runs"] / stats["pressure_matches"],
        np.nan,
    )

    logger.info(
        "compute_batter_pressure_stats: core pressure stats done for %d batters.",
        len(stats),
    )

    # ------------------------------------------------------------------ #
    # IPL vs T20I pressure splits
    # ------------------------------------------------------------------ #
    def _pressure_format_split(fmt_label: str, prefix: str) -> pd.DataFrame:
        """Compute pressure stats for a single match format."""
        if "match_format" not in pressure.columns:
            logger.warning(
                "compute_batter_pressure_stats: 'match_format' column missing – "
                "%s pressure split will be NaN.", fmt_label
            )
            return pd.DataFrame(
                columns=[
                    f"{prefix}_pressure_balls", f"{prefix}_pressure_runs",
                    f"{prefix}_pressure_sr", f"{prefix}_pressure_matches",
                ],
                index=stats.index,
                dtype=float,
            )

        sub = pressure[pressure["match_format"].astype(str).str.upper() == fmt_label.upper()]
        if sub.empty:
            return pd.DataFrame(
                columns=[
                    f"{prefix}_pressure_balls", f"{prefix}_pressure_runs",
                    f"{prefix}_pressure_sr", f"{prefix}_pressure_matches",
                ],
                index=stats.index,
                dtype=float,
            )

        sg = sub.groupby("striker_id", sort=True)
        balls   = sg["is_legal_ball"].count().rename(f"{prefix}_pressure_balls")
        runs    = sg["runs_off_bat"].sum().rename(f"{prefix}_pressure_runs")
        matches = sg["match_id"].nunique().rename(f"{prefix}_pressure_matches")
        split   = pd.concat([balls, runs, matches], axis=1)
        split[f"{prefix}_pressure_sr"] = np.where(
            split[f"{prefix}_pressure_balls"] > 0,
            split[f"{prefix}_pressure_runs"] / split[f"{prefix}_pressure_balls"] * 100.0,
            np.nan,
        )
        return split

    ipl_split  = _pressure_format_split("T20",  "ipl")
    t20i_split = _pressure_format_split("T20I", "t20i")

    stats = stats.join(ipl_split,  how="left")
    stats = stats.join(t20i_split, how="left")

    # Fill integer count columns with 0 where NaN (player had no data for that format)
    split_count_cols = [
        c for c in stats.columns
        if c in {
            "ipl_pressure_balls", "ipl_pressure_runs", "ipl_pressure_matches",
            "t20i_pressure_balls", "t20i_pressure_runs", "t20i_pressure_matches",
        }
    ]
    stats[split_count_cols] = stats[split_count_cols].fillna(0).astype(int)

    # ------------------------------------------------------------------ #
    # Overall total_balls (from baselines if available; else recompute)
    # ------------------------------------------------------------------ #
    if batter_baselines_df is not None and not batter_baselines_df.empty:
        if "total_balls_faced" in batter_baselines_df.columns:
            stats["total_balls"] = (
                batter_baselines_df["total_balls_faced"].reindex(stats.index)
            )
        else:
            logger.warning(
                "compute_batter_pressure_stats: batter_baselines_df has no "
                "'total_balls_faced' column – recomputing total_balls."
            )
            stats["total_balls"] = (
                legal.groupby("striker_id")["is_legal_ball"].count().reindex(stats.index)
            )
    else:
        stats["total_balls"] = (
            legal.groupby("striker_id")["is_legal_ball"].count().reindex(stats.index)
        )

    # ------------------------------------------------------------------ #
    # Main format (the format a player appears in most often)
    # ------------------------------------------------------------------ #
    if "match_format" in legal.columns:
        main_fmt = (
            legal.groupby("striker_id")["match_format"]
            .agg(_most_common)
            .rename("main_team_format")
        )
        stats = stats.join(main_fmt, how="left")
    else:
        stats["main_team_format"] = pd.NA

    # ------------------------------------------------------------------ #
    # Baseline comparison (requires batter_baselines_df)
    # ------------------------------------------------------------------ #
    if batter_baselines_df is not None and not batter_baselines_df.empty:
        logger.info("compute_batter_pressure_stats: merging batter baselines.")
        bl = batter_baselines_df.reindex(stats.index)

        stats["baseline_sr"] = bl.get("overall_strike_rate", pd.Series(dtype=float))
        stats["normal_boundary_pct"] = bl.get("normal_boundary_pct", pd.Series(dtype=float))
        stats["normal_dot_pct"] = bl.get("dot_ball_pct", pd.Series(dtype=float))

        # Fill player_name / main_team_id from baselines for any IDs that
        # didn't appear in the pressure subset (edge case)
        if "player_name" in bl.columns:
            stats["player_name"] = stats["player_name"].where(
                stats["player_name"].notna(), bl["player_name"]
            )

        stats["sr_improvement"] = stats["pressure_strike_rate"] - stats["baseline_sr"]
        stats["boundary_improvement"] = stats["pressure_boundary_pct"] - stats["normal_boundary_pct"]
        stats["dot_reduction"] = stats["normal_dot_pct"] - stats["pressure_dot_pct"]

        logger.info("compute_batter_pressure_stats: baseline comparison columns added.")
    else:
        logger.info(
            "compute_batter_pressure_stats: no batter_baselines_df supplied – "
            "baseline comparison columns will be NaN."
        )
        stats["baseline_sr"]           = np.nan
        stats["normal_boundary_pct"]   = np.nan
        stats["normal_dot_pct"]        = np.nan
        stats["sr_improvement"]        = np.nan
        stats["boundary_improvement"]  = np.nan
        stats["dot_reduction"]         = np.nan

    # ------------------------------------------------------------------ #
    # Eligibility flag
    # ------------------------------------------------------------------ #
    stats["is_eligible"] = (
        (stats["total_balls"].fillna(0)       >= ELIGIBILITY_THRESHOLDS["min_total_balls"])
        & (stats["pressure_balls"].fillna(0)  >= ELIGIBILITY_THRESHOLDS["min_pressure_balls"])
        & (stats["pressure_matches"].fillna(0) >= ELIGIBILITY_THRESHOLDS["min_pressure_matches"])
    )

    eligible_count = int(stats["is_eligible"].sum())
    logger.info(
        "compute_batter_pressure_stats: %d / %d batters are eligible.",
        eligible_count, len(stats),
    )

    # ------------------------------------------------------------------ #
    # Final column ordering for clean export
    # ------------------------------------------------------------------ #
    ordered_cols = [
        # Identity
        "player_name", "main_team_id", "main_team_format",
        # Overall
        "total_balls",
        # Pressure core
        "pressure_balls", "pressure_runs", "pressure_strike_rate",
        "pressure_boundaries", "pressure_boundary_pct",
        "pressure_dots", "pressure_dot_pct",
        "pressure_wickets", "pressure_dismissal_rate",
        "pressure_matches", "pressure_runs_per_match",
        "avg_bpi",
        # Expected performance
        "expected_pressure_runs", "runs_above_expected_pressure",
        # Format splits
        "ipl_pressure_balls", "ipl_pressure_runs", "ipl_pressure_sr", "ipl_pressure_matches",
        "t20i_pressure_balls", "t20i_pressure_runs", "t20i_pressure_sr", "t20i_pressure_matches",
        # Baseline comparison
        "baseline_sr", "sr_improvement",
        "normal_boundary_pct", "boundary_improvement",
        "normal_dot_pct", "dot_reduction",
        # Eligibility
        "is_eligible",
    ]
    ordered_cols = [c for c in ordered_cols if c in stats.columns]
    # Append any extra columns we didn't anticipate
    extra_cols = [c for c in stats.columns if c not in ordered_cols]
    result = stats[ordered_cols + extra_cols].copy()
    result.index.name = "striker_id"

    logger.info(
        "compute_batter_pressure_stats: complete – %d batters, %d eligible.",
        len(result), eligible_count,
    )
    return result


def get_eligible_batters(
    df: pd.DataFrame,
    batter_baselines_df: pd.DataFrame | None = None,
    thresholds: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Return only batters who meet the eligibility thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Full enriched ball-by-ball dataframe.
    batter_baselines_df : pd.DataFrame, optional
        Batter baseline frame from ``player_baseline.compute_batter_baselines``.
    thresholds : dict, optional
        Override dict with keys ``min_total_balls``, ``min_pressure_balls``,
        ``min_pressure_matches``.  Defaults to ``ELIGIBILITY_THRESHOLDS``.

    Returns
    -------
    pd.DataFrame
        Filtered stats frame containing only eligible batters.
    """
    effective = {**ELIGIBILITY_THRESHOLDS, **(thresholds or {})}
    logger.info(
        "get_eligible_batters: thresholds = total_balls>=%d, "
        "pressure_balls>=%d, pressure_matches>=%d.",
        effective["min_total_balls"],
        effective["min_pressure_balls"],
        effective["min_pressure_matches"],
    )

    stats = compute_batter_pressure_stats(df, batter_baselines_df)
    if stats.empty:
        logger.warning("get_eligible_batters: stats frame is empty – returning empty.")
        return stats

    # Apply potentially overridden thresholds
    mask = (
        (stats["total_balls"].fillna(0)        >= effective["min_total_balls"])
        & (stats["pressure_balls"].fillna(0)   >= effective["min_pressure_balls"])
        & (stats["pressure_matches"].fillna(0) >= effective["min_pressure_matches"])
    )
    eligible = stats[mask].copy()

    logger.info(
        "get_eligible_batters: %d / %d batters pass eligibility.",
        len(eligible), len(stats),
    )
    return eligible


def get_batter_analysis_summary(stats_df: pd.DataFrame) -> dict[str, Any]:
    """Return a high-level summary dict from a batter stats / eligible frame.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output of ``compute_batter_pressure_stats`` or ``get_eligible_batters``.

    Returns
    -------
    dict with keys:
        n_eligible         – number of eligible rows in *stats_df*
        avg_pressure_sr    – mean pressure strike rate across eligible players
        avg_pressure_balls – mean pressure balls faced across eligible players
        top_player_name    – player with highest pressure_strike_rate
        top_player_sr      – that player's pressure_strike_rate
    """
    if stats_df is None or stats_df.empty:
        logger.warning("get_batter_analysis_summary: empty stats_df received.")
        return {
            "n_eligible": 0,
            "avg_pressure_sr": np.nan,
            "avg_pressure_balls": np.nan,
            "top_player_name": None,
            "top_player_sr": np.nan,
        }

    # If 'is_eligible' column present, restrict to eligible rows
    if "is_eligible" in stats_df.columns:
        eligible = stats_df[stats_df["is_eligible"].astype(bool)]
    else:
        eligible = stats_df

    n_eligible        = int(len(eligible))
    avg_pressure_sr   = float(eligible["pressure_strike_rate"].mean())   if "pressure_strike_rate" in eligible.columns else np.nan
    avg_pressure_balls = float(eligible["pressure_balls"].mean())         if "pressure_balls"        in eligible.columns else np.nan

    top_player_name: str | None = None
    top_player_sr: float = np.nan
    if "pressure_strike_rate" in eligible.columns and not eligible.empty:
        top_idx = eligible["pressure_strike_rate"].idxmax()
        top_player_sr = float(eligible.loc[top_idx, "pressure_strike_rate"])
        top_player_name = str(
            eligible.loc[top_idx, "player_name"]
            if "player_name" in eligible.columns
            else top_idx
        )

    summary = {
        "n_eligible": n_eligible,
        "avg_pressure_sr": round(avg_pressure_sr, 2) if not np.isnan(avg_pressure_sr) else np.nan,
        "avg_pressure_balls": round(avg_pressure_balls, 1) if not np.isnan(avg_pressure_balls) else np.nan,
        "top_player_name": top_player_name,
        "top_player_sr": round(top_player_sr, 2) if not np.isnan(top_player_sr) else np.nan,
    }
    logger.info(
        "get_batter_analysis_summary: n_eligible=%d, avg_pressure_sr=%.2f, "
        "top=%s (SR %.2f).",
        summary["n_eligible"],
        summary["avg_pressure_sr"] if not np.isnan(summary["avg_pressure_sr"]) else -1,
        summary["top_player_name"],
        summary["top_player_sr"] if not np.isnan(summary["top_player_sr"]) else -1,
    )
    return summary
