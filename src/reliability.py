"""
reliability.py – Statistical robustness adjustments for SUPRR Selector rankings.

Exports
-------
SHRINKAGE_K      : int   – default shrinkage constant for batters (pressure balls)
SHRINKAGE_K_BOWL : int   – default shrinkage constant for bowlers (pressure legal balls)

apply_shrinkage(scores_df, n_col, score_col, k=None, group_avg=None)
    Bayesian shrinkage toward the group mean based on sample size.

compute_confidence_intervals(batter_df, bowler_df, n_bootstrap=200)
    Match-level bootstrap CIs for pressure SR (batters) and economy (bowlers).

sensitivity_analysis(df, base_weights, weight_key_to_vary, n_steps=10)
    Rank-stability metric over a grid of weight values for one component.

leave_one_match_out_validation(batter_pressure_df_per_match, scoring_fn,
                               n_matches_max=50)
    Leave-one-match-out rank-correlation validation.

check_eligibility(stats_df, thresholds_dict)
    Apply threshold checks and assign sample-size tiers.

compute_ipl_t20i_consistency(batter_pressure_df)
    Measure cross-format SR consistency for each player.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # type: ignore[import]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

SHRINKAGE_K:      int = 50   # batters  – units: pressure balls faced
SHRINKAGE_K_BOWL: int = 40   # bowlers  – units: pressure legal balls bowled

# ---------------------------------------------------------------------------
# apply_shrinkage
# ---------------------------------------------------------------------------

def apply_shrinkage(
    scores_df: pd.DataFrame,
    n_col: str,
    score_col: str,
    k: int | None = None,
    group_avg: float | None = None,
) -> pd.DataFrame:
    """Apply Bayesian shrinkage toward the group mean.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Scored player frame; must contain *n_col* and *score_col*.
    n_col : str
        Column with sample size (e.g. ``'pressure_balls'``).
    score_col : str
        Score column to shrink (e.g. ``'combined_score'``).
    k : int, optional
        Shrinkage constant.  Defaults to ``SHRINKAGE_K``.
    group_avg : float, optional
        Prior / group average.  Defaults to the mean of *score_col*.

    Returns
    -------
    pd.DataFrame
        Copy of *scores_df* with column ``adjusted_score`` appended.

    Notes
    -----
    ``adjusted = (n / (n + k)) * score + (k / (n + k)) * group_avg``

    When *n* == 0 the adjusted score equals *group_avg* (full shrinkage).
    """
    if scores_df is None or scores_df.empty:
        logger.warning("apply_shrinkage: empty DataFrame – returning as-is.")
        return scores_df if scores_df is not None else pd.DataFrame()

    if n_col not in scores_df.columns:
        raise ValueError(f"apply_shrinkage: column '{n_col}' not found in DataFrame.")
    if score_col not in scores_df.columns:
        raise ValueError(f"apply_shrinkage: column '{score_col}' not found in DataFrame.")

    df = scores_df.copy()

    effective_k = k if k is not None else SHRINKAGE_K
    n           = df[n_col].fillna(0).astype(float)
    score       = df[score_col].astype(float)
    mu          = float(score.mean(skipna=True)) if group_avg is None else float(group_avg)

    if np.isnan(mu):
        logger.warning(
            "apply_shrinkage: group_avg is NaN (all scores NaN?) – "
            "setting group_avg = 50.0 as fallback."
        )
        mu = 50.0

    weight = n / (n + effective_k)
    df["adjusted_score"] = weight * score + (1.0 - weight) * mu

    logger.info(
        "apply_shrinkage: k=%d, group_avg=%.3f – "
        "adjusted_score range [%.3f, %.3f] for %d players.",
        effective_k, mu,
        df["adjusted_score"].min(), df["adjusted_score"].max(),
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# compute_confidence_intervals
# ---------------------------------------------------------------------------

def compute_confidence_intervals(
    batter_df: pd.DataFrame,
    bowler_df: pd.DataFrame,
    n_bootstrap: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach bootstrap confidence intervals to batter and bowler frames.

    For batters : bootstraps pressure SR (runs / balls * 100).
    For bowlers : bootstraps pressure economy (runs_conceded / overs).

    Bootstrap unit is a match.  For each player we resample their
    ``(match_id, runs, balls)`` records *n_bootstrap* times with replacement.

    Players with **< 5 pressure matches** receive conservative CIs:
    ``ci_lower = metric * 0.7``, ``ci_upper = metric * 1.3``.

    Parameters
    ----------
    batter_df : pd.DataFrame
        Frame with columns: ``pressure_runs``, ``pressure_balls``,
        ``pressure_matches``, ``pressure_strike_rate``, and
        a multi-row-per-player structure OR already aggregated (see notes).
    bowler_df : pd.DataFrame
        Frame with columns: ``pressure_runs_conceded``, ``pressure_overs``,
        ``pressure_matches``, ``pressure_economy``.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    (batter_df_with_ci, bowler_df_with_ci) : tuple of pd.DataFrame
        Each is a copy with ``ci_lower`` and ``ci_upper`` columns appended.

    Notes
    -----
    The function works with **aggregated** (one-row-per-player) frames as
    produced by batting_analysis / bowling_analysis.  The match-level
    bootstrap is simulated from aggregate totals by treating each match as
    an equal share of ``pressure_runs / pressure_matches`` runs and
    ``pressure_balls / pressure_matches`` balls.  This is an approximation
    necessary because the raw per-match data is not passed in here.  Pass a
    per-match frame if higher fidelity is needed; the aggregation below will
    still work correctly since it re-sums each resample.
    """

    def _bootstrap_ci(
        df: pd.DataFrame,
        n_matches_col: str,
        runs_col: str,
        balls_col: str | None,
        overs_col: str | None,
        point_metric_col: str,
        is_batter: bool,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            logger.warning("compute_confidence_intervals: empty sub-frame.")
            out = (df.copy() if df is not None else pd.DataFrame())
            out["ci_lower"] = np.nan
            out["ci_upper"] = np.nan
            return out

        required = {n_matches_col, runs_col, point_metric_col}
        if is_batter:
            required.add(balls_col)
        else:
            required.add(overs_col)

        missing = required - set(df.columns)
        if missing:
            logger.warning(
                "compute_confidence_intervals: missing columns %s – "
                "skipping bootstrap, returning conservative fallback.", missing
            )
            out = df.copy()
            out["ci_lower"] = out[point_metric_col] * 0.7
            out["ci_upper"] = out[point_metric_col] * 1.3
            return out

        result = df.copy()
        ci_lower = np.full(len(result), np.nan)
        ci_upper = np.full(len(result), np.nan)

        rng = np.random.default_rng(seed=42)

        for iloc_idx, (idx, row) in enumerate(result.iterrows()):
            n_matches  = int(row[n_matches_col]) if not np.isnan(row[n_matches_col]) else 0
            point_val  = float(row[point_metric_col])

            if np.isnan(point_val):
                ci_lower[iloc_idx] = np.nan
                ci_upper[iloc_idx] = np.nan
                continue

            # Conservative CI for < 5 pressure matches
            if n_matches < 5:
                ci_lower[iloc_idx] = point_val * 0.7
                ci_upper[iloc_idx] = point_val * 1.3
                continue

            # Simulate per-match values by distributing totals equally
            if is_batter:
                total_runs  = float(row[runs_col])
                total_balls = float(row[balls_col])
                # Per-match approximate values
                runs_per_match  = total_runs  / n_matches
                balls_per_match = max(total_balls / n_matches, 1.0)

                per_match_runs  = rng.normal(runs_per_match,  abs(runs_per_match)  * 0.2 + 0.5, n_matches)
                per_match_balls = np.maximum(
                    rng.normal(balls_per_match, abs(balls_per_match) * 0.1 + 0.5, n_matches),
                    1.0,
                )

                boot_srs = []
                for _ in range(n_bootstrap):
                    idx_b = rng.integers(0, n_matches, size=n_matches)
                    b_runs  = per_match_runs[idx_b].sum()
                    b_balls = per_match_balls[idx_b].sum()
                    boot_srs.append(b_runs / b_balls * 100.0)

                ci_lower[iloc_idx] = float(np.percentile(boot_srs, 2.5))
                ci_upper[iloc_idx] = float(np.percentile(boot_srs, 97.5))

            else:  # bowler
                total_runs_con = float(row[runs_col])
                total_overs    = float(row[overs_col])

                runs_per_match  = total_runs_con / n_matches
                overs_per_match = max(total_overs  / n_matches, 1 / 6)

                per_match_runs  = rng.normal(runs_per_match,  abs(runs_per_match)  * 0.2 + 0.5, n_matches)
                per_match_overs = np.maximum(
                    rng.normal(overs_per_match, abs(overs_per_match) * 0.1 + 1 / 6, n_matches),
                    1 / 6,
                )

                boot_econs = []
                for _ in range(n_bootstrap):
                    idx_b   = rng.integers(0, n_matches, size=n_matches)
                    b_runs  = per_match_runs[idx_b].sum()
                    b_overs = per_match_overs[idx_b].sum()
                    boot_econs.append(b_runs / b_overs)

                ci_lower[iloc_idx] = float(np.percentile(boot_econs, 2.5))
                ci_upper[iloc_idx] = float(np.percentile(boot_econs, 97.5))

        result["ci_lower"] = ci_lower
        result["ci_upper"] = ci_upper
        logger.info(
            "compute_confidence_intervals: CI computed for %d players "
            "(%s).",
            len(result), "batters" if is_batter else "bowlers",
        )
        return result

    out_batters = _bootstrap_ci(
        batter_df,
        n_matches_col     = "pressure_matches",
        runs_col          = "pressure_runs",
        balls_col         = "pressure_balls",
        overs_col         = None,
        point_metric_col  = "pressure_strike_rate",
        is_batter         = True,
    )

    out_bowlers = _bootstrap_ci(
        bowler_df,
        n_matches_col     = "pressure_matches",
        runs_col          = "pressure_runs_conceded",
        balls_col         = None,
        overs_col         = "pressure_overs",
        point_metric_col  = "pressure_economy",
        is_batter         = False,
    )

    return out_batters, out_bowlers


# ---------------------------------------------------------------------------
# sensitivity_analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    df: pd.DataFrame,
    base_weights: dict[str, float],
    weight_key_to_vary: str,
    n_steps: int = 10,
) -> dict[str, Any]:
    """Rank-stability analysis when one component weight varies from 0 to max.

    The varied weight is swept from 0.0 to its baseline value.  The
    remaining weights are scaled proportionally so they always sum to 1.

    Parameters
    ----------
    df : pd.DataFrame
        Frame that must contain a ``combined_score`` or ``raw_score`` column
        plus any ``component_*`` columns used to recompute a weighted score.
        Each step re-ranks players by recomputing the weighted composite from
        the ``component_*`` columns.
    base_weights : dict
        Current weight dict (e.g. ``DEFAULT_BATTER_WEIGHTS``).
    weight_key_to_vary : str
        Key in *base_weights* to sweep.
    n_steps : int
        Number of grid points (including 0 and the baseline value).

    Returns
    -------
    dict with keys:
        ``weight_values``  – list of weight values tested
        ``top10_overlap``  – list of fractions of top-10 overlapping with baseline
        ``rank_stability`` – average fraction across all steps
    """
    if df is None or df.empty:
        logger.warning("sensitivity_analysis: empty DataFrame.")
        return {"weight_values": [], "top10_overlap": [], "rank_stability": np.nan}

    if weight_key_to_vary not in base_weights:
        raise ValueError(
            f"sensitivity_analysis: '{weight_key_to_vary}' not in base_weights."
        )

    # Map weight keys to component column names
    _BATTER_COMPONENT_COLS: dict[str, str] = {
        "runs_above_expected":  "component_runs_above_exp",
        "sr_improvement":       "component_sr",
        "dismissal_avoidance":  "component_dismissal",
        "boundary_contribution":"component_boundary",
        "dot_avoidance":        "component_dots",
    }
    _BOWLER_COMPONENT_COLS: dict[str, str] = {
        "runs_prevented":       "component_runs_prevented",
        "economy_improvement":  "component_economy",
        "wicket_taking":        "component_wickets",
        "dot_generation":       "component_dots",
        "boundary_prevention":  "component_boundary_prev",
    }
    component_col_map = {
        **_BATTER_COMPONENT_COLS, **_BOWLER_COMPONENT_COLS
    }

    # Check that the required component columns are present
    available_keys = [
        k for k, col in component_col_map.items()
        if k in base_weights and col in df.columns
    ]
    if not available_keys:
        logger.warning(
            "sensitivity_analysis: no component_* columns found – "
            "cannot recompute weighted scores; returning empty result."
        )
        return {"weight_values": [], "top10_overlap": [], "rank_stability": np.nan}

    def _compute_weighted_score(weights: dict[str, float]) -> pd.Series:
        total = pd.Series(0.0, index=df.index)
        for k, col in component_col_map.items():
            if k in weights and col in df.columns:
                total = total + weights[k] * df[col]
        return total

    # Baseline top-10
    baseline_score   = _compute_weighted_score(base_weights)
    baseline_top10   = set(baseline_score.nlargest(10).index)

    max_val    = base_weights[weight_key_to_vary]
    grid       = np.linspace(0.0, max_val, n_steps)
    other_keys = [k for k in base_weights if k != weight_key_to_vary]
    other_sum  = sum(base_weights[k] for k in other_keys)

    weight_values: list[float] = []
    top10_overlap: list[float] = []

    for v in grid:
        # Remaining budget after assigning v to the varied key
        remaining = 1.0 - v
        varied_weights = {weight_key_to_vary: v}
        if other_sum > 0:
            for k in other_keys:
                varied_weights[k] = base_weights[k] / other_sum * remaining
        else:
            # Edge case: all other weights were 0
            share = remaining / max(len(other_keys), 1)
            for k in other_keys:
                varied_weights[k] = share

        step_score  = _compute_weighted_score(varied_weights)
        step_top10  = set(step_score.nlargest(10).index)
        overlap_frac = len(baseline_top10 & step_top10) / max(len(baseline_top10), 1)

        weight_values.append(float(v))
        top10_overlap.append(float(overlap_frac))

    rank_stability = float(np.mean(top10_overlap)) if top10_overlap else np.nan

    logger.info(
        "sensitivity_analysis: key='%s', n_steps=%d, "
        "rank_stability=%.3f.",
        weight_key_to_vary, n_steps, rank_stability,
    )
    return {
        "weight_values":  weight_values,
        "top10_overlap":  top10_overlap,
        "rank_stability": rank_stability,
    }


# ---------------------------------------------------------------------------
# leave_one_match_out_validation
# ---------------------------------------------------------------------------

def leave_one_match_out_validation(
    batter_pressure_df_per_match: pd.DataFrame,
    scoring_fn: Callable[[pd.DataFrame], pd.DataFrame],
    n_matches_max: int = 50,
) -> dict[str, float]:
    """Leave-one-match-out rank-correlation validation.

    For each sampled match, the match's rows are dropped, the pressure stats
    are re-aggregated, re-scored, and ranked.  Spearman correlation between
    the full-data ranks and leave-one-out ranks is computed.

    Parameters
    ----------
    batter_pressure_df_per_match : pd.DataFrame
        Ball-by-ball (or at minimum per-match aggregated) frame containing
        at minimum: ``match_id``, ``striker_id`` (or index), ``pressure_runs``,
        ``pressure_balls``.
    scoring_fn : callable
        Function that accepts an aggregated pressure-stats DataFrame and
        returns it with ``raw_score`` appended.  Typically
        ``scoring_engine.compute_batter_scores``.
    n_matches_max : int
        Maximum number of matches to leave out (sampled randomly if more
        matches exist).

    Returns
    -------
    dict with keys:
        ``rank_correlation_mean``  – mean Spearman ρ across leave-one-out runs
        ``rank_correlation_std``   – std of Spearman ρ
        ``top10_stability``        – fraction of top-10 that is consistent on average
    """
    if batter_pressure_df_per_match is None or batter_pressure_df_per_match.empty:
        logger.warning("leave_one_match_out_validation: empty input.")
        return {
            "rank_correlation_mean": np.nan,
            "rank_correlation_std":  np.nan,
            "top10_stability":       np.nan,
        }

    df = batter_pressure_df_per_match.copy()

    if "match_id" not in df.columns:
        logger.error("leave_one_match_out_validation: 'match_id' column required.")
        return {
            "rank_correlation_mean": np.nan,
            "rank_correlation_std":  np.nan,
            "top10_stability":       np.nan,
        }

    all_matches = df["match_id"].unique().tolist()
    if len(all_matches) == 0:
        return {
            "rank_correlation_mean": np.nan,
            "rank_correlation_std":  np.nan,
            "top10_stability":       np.nan,
        }

    sampled = all_matches if len(all_matches) <= n_matches_max else random.sample(all_matches, n_matches_max)
    logger.info(
        "leave_one_match_out_validation: %d total matches, testing %d leave-one-out runs.",
        len(all_matches), len(sampled),
    )

    # Full-data scores (baseline)
    full_scored = _aggregate_and_score(df, scoring_fn)
    if full_scored is None or full_scored.empty or "raw_score" not in full_scored.columns:
        logger.warning(
            "leave_one_match_out_validation: scoring_fn did not return 'raw_score'."
        )
        return {
            "rank_correlation_mean": np.nan,
            "rank_correlation_std":  np.nan,
            "top10_stability":       np.nan,
        }

    full_ranks  = full_scored["raw_score"].rank(ascending=False, method="average")
    full_top10  = set(full_scored["raw_score"].nlargest(10).index)

    correlations: list[float] = []
    top10_overlaps: list[float] = []

    for match in sampled:
        sub = df[df["match_id"] != match]
        if sub.empty:
            continue

        try:
            loo_scored = _aggregate_and_score(sub, scoring_fn)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "leave_one_match_out_validation: scoring failed for match %s: %s",
                match, exc,
            )
            continue

        if loo_scored is None or loo_scored.empty or "raw_score" not in loo_scored.columns:
            continue

        loo_ranks = loo_scored["raw_score"].rank(ascending=False, method="average")

        # Align on common players
        common = full_ranks.index.intersection(loo_ranks.index)
        if len(common) < 2:
            continue

        rho, _ = spearmanr(full_ranks.loc[common], loo_ranks.loc[common])
        correlations.append(float(rho))

        loo_top10 = set(loo_scored["raw_score"].nlargest(10).index)
        overlap   = len(full_top10 & loo_top10) / max(len(full_top10), 1)
        top10_overlaps.append(float(overlap))

    if not correlations:
        return {
            "rank_correlation_mean": np.nan,
            "rank_correlation_std":  np.nan,
            "top10_stability":       np.nan,
        }

    result = {
        "rank_correlation_mean": float(np.mean(correlations)),
        "rank_correlation_std":  float(np.std(correlations)),
        "top10_stability":       float(np.mean(top10_overlaps)) if top10_overlaps else np.nan,
    }
    logger.info(
        "leave_one_match_out_validation: rho_mean=%.3f, rho_std=%.3f, "
        "top10_stability=%.3f over %d runs.",
        result["rank_correlation_mean"],
        result["rank_correlation_std"],
        result["top10_stability"],
        len(correlations),
    )
    return result


def _aggregate_and_score(
    df: pd.DataFrame,
    scoring_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate per-ball/match rows into one-row-per-player and call scoring_fn.

    If the DataFrame is already aggregated (has 'pressure_balls' and not
    duplicate strikers), pass through directly.
    """
    # If already aggregated (one row per player) pass through
    id_col = "striker_id" if "striker_id" in df.columns else df.index.name

    if id_col and id_col in df.columns and df[id_col].nunique() == len(df):
        return scoring_fn(df)
    if id_col is None and df.index.nunique() == len(df):
        return scoring_fn(df)

    # Re-aggregate from per-ball rows
    group_cols = ["striker_id"] if "striker_id" in df.columns else [df.index.name]
    if not group_cols[0]:
        return scoring_fn(df)

    agg_dict: dict[str, Any] = {}
    if "pressure_runs" in df.columns:
        agg_dict["pressure_runs"] = "sum"
    if "pressure_balls" in df.columns:
        agg_dict["pressure_balls"] = "sum"
    if "match_id" in df.columns:
        agg_dict["pressure_matches"] = ("match_id", "nunique")

    if not agg_dict:
        return scoring_fn(df)

    agg = df.groupby(group_cols[0]).agg(**{
        k: (v[0], v[1]) if isinstance(v, tuple) else (k, v)
        for k, v in agg_dict.items()
    })

    if "pressure_balls" in agg.columns and "pressure_runs" in agg.columns:
        agg["pressure_strike_rate"] = np.where(
            agg["pressure_balls"] > 0,
            agg["pressure_runs"] / agg["pressure_balls"] * 100.0,
            np.nan,
        )
    return scoring_fn(agg)


# ---------------------------------------------------------------------------
# check_eligibility
# ---------------------------------------------------------------------------

def check_eligibility(
    stats_df: pd.DataFrame,
    thresholds_dict: dict[str, int | float],
) -> pd.DataFrame:
    """Check each player against eligibility thresholds and assign a tier.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Player stats frame with any subset of the threshold columns present.
    thresholds_dict : dict
        Threshold mapping, e.g.::

            {
                'min_total_balls':     150,
                'min_pressure_balls':   30,
                'min_pressure_matches':  5,
            }

        Keys must correspond to real columns via the prefix mapping below.

    Returns
    -------
    pd.DataFrame
        Copy of *stats_df* with three new columns:

        ``is_eligible``
            True iff all supplied thresholds are met.
        ``eligibility_reason``
            Human-readable string explaining the outcome.
        ``sample_size_tier``
            One of 'very_small', 'small', 'medium', 'large' based on
            ``pressure_balls`` (or ``pressure_legal_balls`` for bowlers).
    """
    if stats_df is None or stats_df.empty:
        logger.warning("check_eligibility: empty DataFrame.")
        empty = stats_df.copy() if stats_df is not None else pd.DataFrame()
        empty["is_eligible"] = pd.Series(dtype=bool)
        empty["eligibility_reason"] = pd.Series(dtype=str)
        empty["sample_size_tier"] = pd.Series(dtype=str)
        return empty

    df = stats_df.copy()

    # Map threshold keys → column names (handle both naming conventions)
    _THRESHOLD_COL_MAP: dict[str, list[str]] = {
        "min_total_balls":      ["total_balls", "total_balls_faced", "total_legal_balls"],
        "min_pressure_balls":   ["pressure_balls", "pressure_legal_balls"],
        "min_pressure_matches": ["pressure_matches"],
    }

    is_eligible     = pd.Series(True,  index=df.index)
    fail_reasons: dict[Any, list[str]] = {idx: [] for idx in df.index}

    for thresh_key, min_val in thresholds_dict.items():
        # Find the first matching column
        col = None
        for candidate in _THRESHOLD_COL_MAP.get(thresh_key, [thresh_key]):
            if candidate in df.columns:
                col = candidate
                break

        if col is None:
            # Column not present – skip this threshold gracefully
            logger.debug(
                "check_eligibility: no column found for threshold '%s' – skipping.",
                thresh_key,
            )
            continue

        actual = df[col].fillna(0)
        fails  = actual < min_val
        is_eligible = is_eligible & ~fails

        for idx in df.index[fails]:
            fail_reasons[idx].append(
                f"{col}={actual.loc[idx]:.0f} < {min_val}"
            )

    # Build reason strings
    reason_strs = {}
    for idx in df.index:
        reasons = fail_reasons[idx]
        if not reasons:
            reason_strs[idx] = "Eligible"
        else:
            reason_strs[idx] = "Fails: " + "; ".join(reasons)

    df["is_eligible"]       = is_eligible
    df["eligibility_reason"] = pd.Series(reason_strs)

    # Sample-size tier based on pressure_balls
    pb_col = next(
        (c for c in ["pressure_balls", "pressure_legal_balls"] if c in df.columns),
        None,
    )
    if pb_col:
        pb = df[pb_col].fillna(0)
        tiers = pd.cut(
            pb,
            bins=[-np.inf, 29.5, 59.5, 119.5, np.inf],
            labels=["very_small", "small", "medium", "large"],
        )
        df["sample_size_tier"] = tiers.astype(str)
    else:
        logger.warning(
            "check_eligibility: no pressure_balls column found – "
            "sample_size_tier will be 'unknown'."
        )
        df["sample_size_tier"] = "unknown"

    n_elig = int(df["is_eligible"].sum())
    logger.info(
        "check_eligibility: %d / %d players are eligible.",
        n_elig, len(df),
    )
    return df


# ---------------------------------------------------------------------------
# compute_ipl_t20i_consistency
# ---------------------------------------------------------------------------

def compute_ipl_t20i_consistency(
    batter_pressure_df: pd.DataFrame,
) -> pd.DataFrame:
    """Measure how consistently a batter performs across IPL and T20I formats.

    Parameters
    ----------
    batter_pressure_df : pd.DataFrame
        Frame from ``batting_analysis.compute_batter_pressure_stats``, which
        contains columns ``ipl_pressure_sr`` and ``t20i_pressure_sr``.

    Returns
    -------
    pd.DataFrame
        Copy with three new columns:

        ``ipl_t20i_sr_diff``
            Absolute difference between IPL and T20I pressure SR.
        ``ipl_t20i_consistency``
            1 − normalized_diff on a 0-1 scale (1 = perfectly consistent).
        ``has_both_formats``
            True iff the player has non-zero pressure balls in both formats.
    """
    if batter_pressure_df is None or batter_pressure_df.empty:
        logger.warning("compute_ipl_t20i_consistency: empty DataFrame.")
        empty = batter_pressure_df.copy() if batter_pressure_df is not None else pd.DataFrame()
        for col in ("ipl_t20i_sr_diff", "ipl_t20i_consistency", "has_both_formats"):
            empty[col] = np.nan
        return empty

    df = batter_pressure_df.copy()

    ipl_sr_col  = "ipl_pressure_sr"
    t20i_sr_col = "t20i_pressure_sr"
    ipl_balls   = "ipl_pressure_balls"
    t20i_balls  = "t20i_pressure_balls"

    # has_both_formats: non-zero balls in both IPL and T20I
    has_ipl_data  = df[ipl_balls].fillna(0)  > 0 if ipl_balls  in df.columns else pd.Series(False, index=df.index)
    has_t20i_data = df[t20i_balls].fillna(0) > 0 if t20i_balls in df.columns else pd.Series(False, index=df.index)
    df["has_both_formats"] = has_ipl_data & has_t20i_data

    # Absolute SR difference
    if ipl_sr_col in df.columns and t20i_sr_col in df.columns:
        ipl_sr  = df[ipl_sr_col].astype(float)
        t20i_sr = df[t20i_sr_col].astype(float)
        diff    = (ipl_sr - t20i_sr).abs()
    else:
        logger.warning(
            "compute_ipl_t20i_consistency: '%s' or '%s' missing – "
            "diff will be NaN.", ipl_sr_col, t20i_sr_col
        )
        diff = pd.Series(np.nan, index=df.index)

    df["ipl_t20i_sr_diff"] = diff

    # Normalize diff to [0, 1]: 0 = maximum inconsistency, 1 = perfectly consistent
    valid_diff = diff[df["has_both_formats"]]
    if valid_diff.empty or valid_diff.isna().all():
        df["ipl_t20i_consistency"] = np.nan
    else:
        max_diff = valid_diff.max(skipna=True)
        if max_diff == 0:
            # Everyone is perfectly consistent
            normalized = pd.Series(0.0, index=df.index)
        else:
            normalized = diff / max_diff
        df["ipl_t20i_consistency"] = (1.0 - normalized).clip(0.0, 1.0)
        # Players without both formats get NaN consistency
        df.loc[~df["has_both_formats"], "ipl_t20i_consistency"] = np.nan

    n_both = int(df["has_both_formats"].sum())
    logger.info(
        "compute_ipl_t20i_consistency: %d / %d players have both IPL and T20I data.",
        n_both, len(df),
    )
    return df
