"""
ranking_engine.py – Master orchestration layer for the SUPRR Selector pipeline.

Exports
-------
run_full_pipeline(bpi_weights, bopi_weights, batter_weights, bowler_weights,
                  eligibility_overrides, force_refit_models)
    Run all pipeline steps end-to-end and return a results dict.

get_batter_profile(player_id, pipeline_results)
    Return a full profile dict for a single batter (for the dashboard).

get_bowler_profile(player_id, pipeline_results)
    Return a full profile dict for a single bowler.

save_outputs(pipeline_results, output_dir)
    Write batter_rankings.csv, bowler_rankings.csv, analysis_summary.json.

Pipeline steps
--------------
1.  Load data         – data_loader.load_enriched_bbb
2.  Clean             – data_cleaning.clean_bbb
3.  Match states      – match_state.compute_match_states
4.  Pressure          – pressure_engine.compute_pressure
5.  Baselines         – player_baseline.compute_batter_baselines / compute_bowler_baselines
6.  Expected models   – expected_performance.load_or_fit_models
7.  Expected columns  – expected_performance.add_expected_columns
8.  Batter stats      – batting_analysis.compute_batter_pressure_stats
9.  Bowler stats      – bowling_analysis.compute_bowler_pressure_stats
10. Eligibility       – get_eligible_batters / get_eligible_bowlers
11. Scores            – scoring_engine.compute_batter_scores / compute_bowler_scores
12. Shrinkage         – reliability.apply_shrinkage
13. CIs               – reliability.compute_confidence_intervals
14. IPL/T20I consist. – reliability.compute_ipl_t20i_consistency
15. Team names        – most-common team from raw data
16. Rank              – sort by adjusted_score, add rank col
17. format_coverage   – 'IPL Only' / 'T20I Only' / 'Both'
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports to allow each step to fail independently
# ---------------------------------------------------------------------------

def _import_step(name: str):
    """Import a src module by name, returning None on failure."""
    import importlib
    import sys
    try:
        return importlib.import_module(name)
    except Exception as exc:
        logger.error("Failed to import module '%s': %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Output dir for model files
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "outputs" / "models"

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self, label: str):
        self.label = label
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        logger.info("[START] %s", self.label)
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        logger.info("[DONE ] %s  (%.2fs)", self.label, elapsed)


# ---------------------------------------------------------------------------
# Team-name helper
# ---------------------------------------------------------------------------

def _resolve_team_names(df: pd.DataFrame, team_id_col: str) -> pd.Series:
    """Return a Series of team names from the teams.csv, keyed by team_id.

    Falls back to str(team_id) for any ID not found.
    """
    try:
        dl = _import_step("data_loader")
        if dl is None:
            return df[team_id_col].astype(str)
        raw = dl.load_all_data()
        teams = raw.get("teams", pd.DataFrame())
        if teams.empty or "team_id" not in teams.columns or "team_name" not in teams.columns:
            return df[team_id_col].astype(str)
        team_map = dict(zip(
            teams["team_id"].astype(str),
            teams["team_name"].astype(str),
        ))
        return df[team_id_col].astype(str).map(team_map).fillna(df[team_id_col].astype(str))
    except Exception as exc:
        logger.warning("_resolve_team_names failed: %s", exc)
        return df[team_id_col].astype(str)


def _add_format_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'format_coverage' column based on IPL / T20I pressure ball counts."""
    df = df.copy()
    ipl_balls  = df.get("ipl_pressure_balls",  pd.Series(0, index=df.index)).fillna(0)
    t20i_balls = df.get("t20i_pressure_balls", pd.Series(0, index=df.index)).fillna(0)

    has_ipl  = ipl_balls  > 0
    has_t20i = t20i_balls > 0

    df["format_coverage"] = np.select(
        [has_ipl & has_t20i, has_ipl & ~has_t20i, ~has_ipl & has_t20i],
        ["Both", "IPL Only", "T20I Only"],
        default="Unknown",
    )
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    bpi_weights: dict | None = None,
    bopi_weights: dict | None = None,
    batter_weights: dict | None = None,
    bowler_weights: dict | None = None,
    eligibility_overrides: dict | None = None,
    force_refit_models: bool = False,
) -> dict[str, Any]:
    """Run the complete SUPRR ranking pipeline.

    Parameters
    ----------
    bpi_weights, bopi_weights : dict, optional
        Override BPI / BoPI component weights (merged with defaults).
    batter_weights, bowler_weights : dict, optional
        Override scoring weights (merged with defaults).
    eligibility_overrides : dict, optional
        Override eligibility thresholds for both batters and bowlers.
    force_refit_models : bool
        Force re-training of expected-performance models even if saved files exist.

    Returns
    -------
    dict with keys:
        batter_rankings, bowler_rankings, batter_full, bowler_full,
        data_summary, batter_baselines, bowler_baselines, full_df
    """
    results: dict[str, Any] = {
        "batter_rankings": pd.DataFrame(),
        "bowler_rankings": pd.DataFrame(),
        "batter_full": pd.DataFrame(),
        "bowler_full": pd.DataFrame(),
        "data_summary": {},
        "batter_baselines": pd.DataFrame(),
        "bowler_baselines": pd.DataFrame(),
        "full_df": pd.DataFrame(),
    }

    # ------------------------------------------------------------------ #
    # Step 1 – Load enriched BBB data
    # ------------------------------------------------------------------ #
    df = None
    with _Timer("Step 1: load_enriched_bbb"):
        try:
            dl = _import_step("data_loader")
            if dl is None:
                raise ImportError("data_loader unavailable")
            df = dl.load_enriched_bbb()
            logger.info("Loaded BBB: %d rows × %d cols", *df.shape)
        except Exception as exc:
            logger.error("Step 1 FAILED: %s", exc, exc_info=True)
            return results

    # ------------------------------------------------------------------ #
    # Step 2 – Clean
    # ------------------------------------------------------------------ #
    with _Timer("Step 2: clean_bbb"):
        try:
            dc = _import_step("data_cleaning")
            if dc is None:
                raise ImportError("data_cleaning unavailable")
            df = dc.clean_bbb(df)
            logger.info("After clean: %d rows", len(df))
        except Exception as exc:
            logger.error("Step 2 FAILED: %s", exc, exc_info=True)
            return results

    # ------------------------------------------------------------------ #
    # Step 3 – Match states
    # ------------------------------------------------------------------ #
    with _Timer("Step 3: compute_match_states"):
        try:
            ms_mod = _import_step("match_state")
            if ms_mod is None:
                raise ImportError("match_state unavailable")
            df = ms_mod.compute_match_states(df)
            logger.info("Match states added: %d cols", df.shape[1])
        except Exception as exc:
            logger.error("Step 3 FAILED: %s", exc, exc_info=True)
            return results

    # ------------------------------------------------------------------ #
    # Step 4 – Pressure
    # ------------------------------------------------------------------ #
    with _Timer("Step 4: compute_pressure"):
        try:
            pe = _import_step("pressure_engine")
            if pe is None:
                raise ImportError("pressure_engine unavailable")
            df = pe.compute_pressure(df, bpi_weights=bpi_weights, bopi_weights=bopi_weights)
            logger.info("Pressure computed: BPI mean=%.2f, BoPI mean=%.2f",
                        df["bpi"].mean(), df["bopi"].mean())
        except Exception as exc:
            logger.error("Step 4 FAILED: %s", exc, exc_info=True)
            return results

    # ------------------------------------------------------------------ #
    # Step 5 – Baselines
    # ------------------------------------------------------------------ #
    batter_baselines = pd.DataFrame()
    bowler_baselines = pd.DataFrame()
    with _Timer("Step 5: compute baselines"):
        try:
            pb = _import_step("player_baseline")
            if pb is None:
                raise ImportError("player_baseline unavailable")
            batter_baselines = pb.compute_batter_baselines(df)
            bowler_baselines = pb.compute_bowler_baselines(df)
            logger.info("Baselines: %d batters, %d bowlers",
                        len(batter_baselines), len(bowler_baselines))
            results["batter_baselines"] = batter_baselines
            results["bowler_baselines"] = bowler_baselines
        except Exception as exc:
            logger.error("Step 5 FAILED: %s – continuing without baselines", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Step 6 – Load or fit expected performance models
    # ------------------------------------------------------------------ #
    batting_model = bowling_model = wicket_model = None
    model_dir = str(_DEFAULT_MODEL_DIR)
    _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if force_refit_models:
        # Remove any existing model files so load_or_fit_models re-trains
        for fname in ("batting_model.joblib", "bowling_model.joblib", "wicket_model.joblib"):
            p = _DEFAULT_MODEL_DIR / fname
            if p.exists():
                p.unlink()
                logger.info("Removed cached model file: %s", p)

    with _Timer("Step 6: load/fit expected performance models"):
        try:
            ep = _import_step("expected_performance")
            if ep is None:
                raise ImportError("expected_performance unavailable")
            batting_model, bowling_model, wicket_model = ep.load_or_fit_models(df, model_dir)
            logger.info("Models ready – batting fitted=%s, bowling fitted=%s, wicket fitted=%s",
                        batting_model._is_fitted, bowling_model._is_fitted, wicket_model._is_fitted)
        except Exception as exc:
            logger.warning("Step 6 FAILED: %s – expected columns will be NaN", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Step 7 – Add expected columns
    # ------------------------------------------------------------------ #
    with _Timer("Step 7: add_expected_columns"):
        try:
            if ep is not None and batting_model is not None:
                df = ep.add_expected_columns(df, batting_model, bowling_model, wicket_model)
                logger.info("Expected columns added.")
            else:
                logger.warning("Step 7 SKIPPED – models not available; expected cols set to NaN.")
                for col in ("expected_runs_per_ball", "expected_runs_conceded_per_ball",
                            "expected_wicket_prob", "runs_above_expected",
                            "runs_prevented", "wicket_above_expected"):
                    df[col] = np.nan
        except Exception as exc:
            logger.warning("Step 7 FAILED: %s – expected cols set to NaN", exc, exc_info=True)
            for col in ("expected_runs_per_ball", "expected_runs_conceded_per_ball",
                        "expected_wicket_prob", "runs_above_expected",
                        "runs_prevented", "wicket_above_expected"):
                if col not in df.columns:
                    df[col] = np.nan

    results["full_df"] = df

    # ------------------------------------------------------------------ #
    # Data summary
    # ------------------------------------------------------------------ #
    try:
        dc_mod = _import_step("data_cleaning")
        if dc_mod is not None and hasattr(dc_mod, "get_data_summary"):
            dl_mod = _import_step("data_loader")
            raw = dl_mod.load_all_data() if dl_mod else {}
            matches_df = raw.get("matches", pd.DataFrame())
            comps_df   = raw.get("competitions", pd.DataFrame())
            data_summary = dc_mod.get_data_summary(df, matches_df, comps_df)
        else:
            data_summary = {
                "total_matches": int(df["match_id"].nunique()),
                "total_deliveries": len(df),
                "total_legal_balls": int(df["is_legal_ball"].astype(bool).sum()),
            }
        results["data_summary"] = data_summary
    except Exception as exc:
        logger.warning("data_summary computation failed: %s", exc)
        results["data_summary"] = {}

    # ------------------------------------------------------------------ #
    # Step 8 – Batter pressure stats
    # ------------------------------------------------------------------ #
    batter_stats = pd.DataFrame()
    with _Timer("Step 8: compute_batter_pressure_stats"):
        try:
            ba = _import_step("batting_analysis")
            if ba is None:
                raise ImportError("batting_analysis unavailable")
            batter_stats = ba.compute_batter_pressure_stats(
                df, batter_baselines_df=batter_baselines
            )
            logger.info("Batter pressure stats: %d batters", len(batter_stats))
        except Exception as exc:
            logger.error("Step 8 FAILED: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Step 9 – Bowler pressure stats
    # ------------------------------------------------------------------ #
    bowler_stats = pd.DataFrame()
    with _Timer("Step 9: compute_bowler_pressure_stats"):
        try:
            bla = _import_step("bowling_analysis")
            if bla is None:
                raise ImportError("bowling_analysis unavailable")
            bowler_stats = bla.compute_bowler_pressure_stats(
                df, bowler_baselines_df=bowler_baselines
            )
            logger.info("Bowler pressure stats: %d bowlers", len(bowler_stats))
        except Exception as exc:
            logger.error("Step 9 FAILED: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Step 10 – Eligibility
    # ------------------------------------------------------------------ #
    eligible_batters = pd.DataFrame()
    eligible_bowlers = pd.DataFrame()
    with _Timer("Step 10: apply eligibility"):
        try:
            if ba is not None and not batter_stats.empty:
                ov = eligibility_overrides or {}
                eligible_batters = ba.get_eligible_batters(
                    df, batter_baselines_df=batter_baselines, thresholds=ov
                )
                logger.info("Eligible batters: %d", len(eligible_batters))
        except Exception as exc:
            logger.error("Batter eligibility FAILED: %s", exc, exc_info=True)

        try:
            if bla is not None and not bowler_stats.empty:
                ov = eligibility_overrides or {}
                eligible_bowlers = bla.get_eligible_bowlers(
                    df, bowler_baselines_df=bowler_baselines, thresholds=ov
                )
                logger.info("Eligible bowlers: %d", len(eligible_bowlers))
        except Exception as exc:
            logger.error("Bowler eligibility FAILED: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Step 11 – Scores
    # ------------------------------------------------------------------ #
    with _Timer("Step 11: compute scores"):
        try:
            se = _import_step("scoring_engine")
            if se is None:
                raise ImportError("scoring_engine unavailable")
            if not eligible_batters.empty:
                eligible_batters = se.compute_batter_scores(eligible_batters, weights=batter_weights)
                logger.info("Batter scores computed.")
            if not eligible_bowlers.empty:
                eligible_bowlers = se.compute_bowler_scores(eligible_bowlers, weights=bowler_weights)
                logger.info("Bowler scores computed.")
        except Exception as exc:
            logger.error("Step 11 FAILED: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Step 12 – Shrinkage
    # ------------------------------------------------------------------ #
    with _Timer("Step 12: apply shrinkage"):
        try:
            rel = _import_step("reliability")
            if rel is None:
                raise ImportError("reliability unavailable")
            if not eligible_batters.empty and "combined_score" in eligible_batters.columns:
                eligible_batters = rel.apply_shrinkage(
                    eligible_batters,
                    n_col="pressure_balls",
                    score_col="combined_score",
                    k=rel.SHRINKAGE_K,
                )
                logger.info("Batter shrinkage applied.")
            else:
                if not eligible_batters.empty:
                    eligible_batters["adjusted_score"] = eligible_batters.get(
                        "combined_score", eligible_batters.get("percentile_score", np.nan)
                    )

            if not eligible_bowlers.empty and "combined_score" in eligible_bowlers.columns:
                eligible_bowlers = rel.apply_shrinkage(
                    eligible_bowlers,
                    n_col="pressure_balls",
                    score_col="combined_score",
                    k=rel.SHRINKAGE_K_BOWL,
                )
                logger.info("Bowler shrinkage applied.")
            else:
                if not eligible_bowlers.empty:
                    eligible_bowlers["adjusted_score"] = eligible_bowlers.get(
                        "combined_score", eligible_bowlers.get("percentile_score", np.nan)
                    )
        except Exception as exc:
            logger.error("Step 12 FAILED: %s", exc, exc_info=True)
            # Fallback: adjusted_score = combined_score
            for frame in [eligible_batters, eligible_bowlers]:
                if not frame.empty and "adjusted_score" not in frame.columns:
                    frame["adjusted_score"] = frame.get(
                        "combined_score", pd.Series(np.nan, index=frame.index)
                    )

    # ------------------------------------------------------------------ #
    # Step 13 – Confidence intervals
    # ------------------------------------------------------------------ #
    with _Timer("Step 13: compute confidence intervals"):
        try:
            if rel is not None and not eligible_batters.empty and not eligible_bowlers.empty:
                eligible_batters, eligible_bowlers = rel.compute_confidence_intervals(
                    eligible_batters, eligible_bowlers
                )
                logger.info("Confidence intervals computed.")
            else:
                for frame in [eligible_batters, eligible_bowlers]:
                    if not frame.empty:
                        frame["ci_lower"] = np.nan
                        frame["ci_upper"] = np.nan
        except Exception as exc:
            logger.warning("Step 13 FAILED: %s – CIs set to NaN", exc, exc_info=True)
            for frame in [eligible_batters, eligible_bowlers]:
                if not frame.empty:
                    frame["ci_lower"] = np.nan
                    frame["ci_upper"] = np.nan

    # ------------------------------------------------------------------ #
    # Step 14 – IPL/T20I consistency
    # ------------------------------------------------------------------ #
    with _Timer("Step 14: compute IPL/T20I consistency"):
        try:
            if rel is not None and not eligible_batters.empty:
                eligible_batters = rel.compute_ipl_t20i_consistency(eligible_batters)
                logger.info("IPL/T20I consistency computed for batters.")
        except Exception as exc:
            logger.warning("Step 14 FAILED: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Step 15 – Add team names
    # ------------------------------------------------------------------ #
    with _Timer("Step 15: resolve team names"):
        try:
            if not eligible_batters.empty and "main_team_id" in eligible_batters.columns:
                eligible_batters["main_team_name"] = _resolve_team_names(
                    eligible_batters, "main_team_id"
                )
            if not eligible_bowlers.empty and "main_team_id" in eligible_bowlers.columns:
                eligible_bowlers["main_team_name"] = _resolve_team_names(
                    eligible_bowlers, "main_team_id"
                )
        except Exception as exc:
            logger.warning("Step 15 FAILED: %s – team name set to team_id", exc, exc_info=True)
            for frame, col in [(eligible_batters, "main_team_id"), (eligible_bowlers, "main_team_id")]:
                if not frame.empty and col in frame.columns:
                    frame["main_team_name"] = frame[col].astype(str)

    # ------------------------------------------------------------------ #
    # Step 16 – Sort by adjusted_score, add rank
    # ------------------------------------------------------------------ #
    with _Timer("Step 16-17: rank and format_coverage"):
        for frame in [eligible_batters, eligible_bowlers]:
            if frame.empty:
                continue
            if "adjusted_score" not in frame.columns:
                frame["adjusted_score"] = np.nan
            frame.sort_values("adjusted_score", ascending=False, inplace=True)
            frame["rank"] = range(1, len(frame) + 1)

        # Step 17: format_coverage
        if not eligible_batters.empty:
            eligible_batters = _add_format_coverage(eligible_batters)
        if not eligible_bowlers.empty:
            eligible_bowlers = _add_format_coverage(eligible_bowlers)

    # ------------------------------------------------------------------ #
    # Store full eligible frames
    # ------------------------------------------------------------------ #
    results["batter_full"] = eligible_batters.copy()
    results["bowler_full"] = eligible_bowlers.copy()

    # ------------------------------------------------------------------ #
    # Build ranking tables (Top 10 with required columns)
    # ------------------------------------------------------------------ #
    _BATTER_RANK_COLS = [
        "rank", "player_name", "main_team_name",
        "adjusted_score", "pressure_runs", "pressure_strike_rate",
        "baseline_sr", "sr_improvement",
        "pressure_balls", "pressure_matches",
        "format_coverage", "ci_lower", "ci_upper",
    ]
    _BOWLER_RANK_COLS = [
        "rank", "player_name", "main_team_name",
        "adjusted_score", "pressure_economy", "baseline_economy",
        "economy_improvement", "pressure_wickets", "wickets_above_expected",
        "pressure_balls", "pressure_matches",
        "format_coverage", "ci_lower", "ci_upper",
    ]

    def _build_ranking_table(frame: pd.DataFrame, cols: list[str], top_n: int = 10) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=cols)
        available = [c for c in cols if c in frame.columns]
        # Ensure all requested cols exist (fill missing with NaN)
        for c in cols:
            if c not in frame.columns:
                frame[c] = np.nan
        top = frame.head(top_n)[cols].copy()
        return top

    results["batter_rankings"] = _build_ranking_table(eligible_batters, _BATTER_RANK_COLS)
    results["bowler_rankings"] = _build_ranking_table(eligible_bowlers, _BOWLER_RANK_COLS)

    logger.info(
        "Pipeline complete. Batter rankings: %d rows, Bowler rankings: %d rows.",
        len(results["batter_rankings"]),
        len(results["bowler_rankings"]),
    )
    return results


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def get_batter_profile(player_id: int, pipeline_results: dict) -> dict:
    """Return a full batter profile dict for use in a dashboard.

    Parameters
    ----------
    player_id : int
        striker_id to look up.
    pipeline_results : dict
        Output of ``run_full_pipeline()``.

    Returns
    -------
    dict  (empty dict if player not found)
    """
    batter_full: pd.DataFrame = pipeline_results.get("batter_full", pd.DataFrame())
    rankings: pd.DataFrame    = pipeline_results.get("batter_rankings", pd.DataFrame())

    if batter_full.empty:
        return {}

    if player_id not in batter_full.index:
        logger.warning("get_batter_profile: player_id %s not found in batter_full.", player_id)
        return {}

    row = batter_full.loc[player_id]

    # Rank from rankings table
    rank_val = None
    if not rankings.empty and "rank" in batter_full.columns:
        rank_val = int(batter_full.loc[player_id, "rank"]) if "rank" in batter_full.columns else None

    def _safe_float(val):
        try:
            v = float(val)
            return None if np.isnan(v) else round(v, 3)
        except (TypeError, ValueError):
            return None

    def _safe_int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    profile = {
        # Identity
        "player_id":       player_id,
        "player_name":     str(row.get("player_name", "")),
        "rank":            rank_val,
        "team":            str(row.get("main_team_name", row.get("main_team_id", ""))),
        "format_coverage": str(row.get("format_coverage", "Unknown")),

        # Adjusted score
        "adjusted_score":  _safe_float(row.get("adjusted_score")),
        "combined_score":  _safe_float(row.get("combined_score")),
        "percentile_score": _safe_float(row.get("percentile_score")),

        # Pressure performance
        "pressure_balls":           _safe_int(row.get("pressure_balls")),
        "pressure_runs":            _safe_int(row.get("pressure_runs")),
        "pressure_matches":         _safe_int(row.get("pressure_matches")),
        "pressure_strike_rate":     _safe_float(row.get("pressure_strike_rate")),
        "pressure_boundary_pct":    _safe_float(row.get("pressure_boundary_pct")),
        "pressure_dot_pct":         _safe_float(row.get("pressure_dot_pct")),
        "pressure_dismissal_rate":  _safe_float(row.get("pressure_dismissal_rate")),
        "avg_bpi":                  _safe_float(row.get("avg_bpi")),
        "runs_above_expected_pressure": _safe_float(row.get("runs_above_expected_pressure")),

        # Baseline comparison
        "baseline_sr":         _safe_float(row.get("baseline_sr")),
        "sr_improvement":      _safe_float(row.get("sr_improvement")),
        "boundary_improvement": _safe_float(row.get("boundary_improvement")),
        "dot_reduction":        _safe_float(row.get("dot_reduction")),

        # Format splits
        "ipl_vs_t20i": {
            "ipl_pressure_balls":   _safe_int(row.get("ipl_pressure_balls")),
            "ipl_pressure_runs":    _safe_int(row.get("ipl_pressure_runs")),
            "ipl_pressure_sr":      _safe_float(row.get("ipl_pressure_sr")),
            "ipl_pressure_matches": _safe_int(row.get("ipl_pressure_matches")),
            "t20i_pressure_balls":  _safe_int(row.get("t20i_pressure_balls")),
            "t20i_pressure_runs":   _safe_int(row.get("t20i_pressure_runs")),
            "t20i_pressure_sr":     _safe_float(row.get("t20i_pressure_sr")),
            "t20i_pressure_matches": _safe_int(row.get("t20i_pressure_matches")),
        },

        # Cross-format consistency
        "ipl_t20i_consistency": _safe_float(row.get("ipl_t20i_consistency")),
        "ipl_t20i_sr_diff":     _safe_float(row.get("ipl_t20i_sr_diff")),

        # Component z-scores
        "component_z_scores": {
            "runs_above_expected": _safe_float(row.get("component_runs_above_exp")),
            "sr_improvement":      _safe_float(row.get("component_sr")),
            "dismissal_avoidance": _safe_float(row.get("component_dismissal")),
            "boundary_contribution": _safe_float(row.get("component_boundary")),
            "dot_avoidance":       _safe_float(row.get("component_dots")),
        },

        # Confidence interval
        "ci_lower": _safe_float(row.get("ci_lower")),
        "ci_upper": _safe_float(row.get("ci_upper")),

        # Narrative
        "why_this_rank": _batter_rank_narrative(row, rank_val),
    }
    return profile


def _batter_rank_narrative(row: pd.Series, rank: int | None) -> dict:
    """Build a human-readable explanation dict for a batter's rank."""
    def _flt(x):
        try:
            v = float(x)
            return None if np.isnan(v) else round(v, 2)
        except (TypeError, ValueError):
            return None

    sr_imp  = _flt(row.get("sr_improvement"))
    psr     = _flt(row.get("pressure_strike_rate"))
    bsr     = _flt(row.get("baseline_sr"))
    pballs  = row.get("pressure_balls", 0)
    rae     = _flt(row.get("runs_above_expected_pressure"))

    parts = []
    if sr_imp is not None:
        direction = "higher" if sr_imp > 0 else "lower"
        parts.append(
            f"Scores {abs(sr_imp):.1f} SR points {direction} under pressure than their baseline."
        )
    if psr is not None:
        parts.append(f"Pressure SR: {psr:.1f}.")
    if rae is not None:
        parts.append(
            f"Scored {rae:+.1f} runs above what the model expected under pressure."
        )

    return {
        "summary": f"Ranked #{rank}." if rank else "Rank not computed.",
        "highlights": parts,
        "stats": {
            "pressure_sr": psr,
            "baseline_sr": bsr,
            "sr_improvement": sr_imp,
            "pressure_balls": int(pballs) if pballs else 0,
            "runs_above_expected": rae,
        },
    }


def get_bowler_profile(player_id: int, pipeline_results: dict) -> dict:
    """Return a full bowler profile dict for use in a dashboard."""
    bowler_full: pd.DataFrame = pipeline_results.get("bowler_full", pd.DataFrame())

    if bowler_full.empty:
        return {}
    if player_id not in bowler_full.index:
        logger.warning("get_bowler_profile: player_id %s not found in bowler_full.", player_id)
        return {}

    row = bowler_full.loc[player_id]

    def _safe_float(val):
        try:
            v = float(val)
            return None if np.isnan(v) else round(v, 3)
        except (TypeError, ValueError):
            return None

    def _safe_int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    rank_val = int(bowler_full.loc[player_id, "rank"]) if "rank" in bowler_full.columns else None

    profile = {
        "player_id":       player_id,
        "player_name":     str(row.get("player_name", "")),
        "rank":            rank_val,
        "team":            str(row.get("main_team_name", row.get("main_team_id", ""))),
        "format_coverage": str(row.get("format_coverage", "Unknown")),

        "adjusted_score":  _safe_float(row.get("adjusted_score")),
        "combined_score":  _safe_float(row.get("combined_score")),
        "percentile_score": _safe_float(row.get("percentile_score")),

        "pressure_balls":          _safe_int(row.get("pressure_balls")),
        "pressure_matches":        _safe_int(row.get("pressure_matches")),
        "pressure_economy":        _safe_float(row.get("pressure_economy")),
        "pressure_wickets":        _safe_int(row.get("pressure_wickets")),
        "pressure_dot_pct":        _safe_float(row.get("pressure_dot_pct")),
        "pressure_boundary_pct":   _safe_float(row.get("pressure_boundary_pct")),
        "avg_bopi":                _safe_float(row.get("avg_bopi")),
        "wickets_above_expected":  _safe_float(row.get("wickets_above_expected")),
        "runs_prevented_pressure": _safe_float(row.get("runs_prevented_pressure")),

        "baseline_economy":    _safe_float(row.get("baseline_economy")),
        "economy_improvement": _safe_float(row.get("economy_improvement")),
        "dot_improvement":     _safe_float(row.get("dot_improvement")),
        "boundary_prevention_improvement": _safe_float(row.get("boundary_prevention_improvement")),

        "ipl_vs_t20i": {
            "ipl_pressure_balls":           _safe_int(row.get("ipl_pressure_balls")),
            "ipl_pressure_runs_conceded":    _safe_int(row.get("ipl_pressure_runs_conceded")),
            "ipl_pressure_economy":         _safe_float(row.get("ipl_pressure_economy")),
            "ipl_pressure_wickets":         _safe_int(row.get("ipl_pressure_wickets")),
            "ipl_pressure_matches":         _safe_int(row.get("ipl_pressure_matches")),
            "t20i_pressure_balls":          _safe_int(row.get("t20i_pressure_balls")),
            "t20i_pressure_runs_conceded":   _safe_int(row.get("t20i_pressure_runs_conceded")),
            "t20i_pressure_economy":        _safe_float(row.get("t20i_pressure_economy")),
            "t20i_pressure_wickets":        _safe_int(row.get("t20i_pressure_wickets")),
            "t20i_pressure_matches":        _safe_int(row.get("t20i_pressure_matches")),
        },

        "component_z_scores": {
            "runs_prevented":    _safe_float(row.get("component_runs_prevented")),
            "economy_improvement": _safe_float(row.get("component_economy")),
            "wicket_taking":     _safe_float(row.get("component_wickets")),
            "dot_generation":    _safe_float(row.get("component_dots")),
            "boundary_prevention": _safe_float(row.get("component_boundary_prev")),
        },

        "ci_lower": _safe_float(row.get("ci_lower")),
        "ci_upper": _safe_float(row.get("ci_upper")),

        "why_this_rank": _bowler_rank_narrative(row, rank_val),
    }
    return profile


def _bowler_rank_narrative(row: pd.Series, rank: int | None) -> dict:
    """Build a human-readable explanation dict for a bowler's rank."""
    def _flt(x):
        try:
            v = float(x)
            return None if np.isnan(v) else round(v, 2)
        except (TypeError, ValueError):
            return None

    econ_imp = _flt(row.get("economy_improvement"))
    peco     = _flt(row.get("pressure_economy"))
    beco     = _flt(row.get("baseline_economy"))
    wae      = _flt(row.get("wickets_above_expected"))
    pballs   = row.get("pressure_balls", 0)

    parts = []
    if econ_imp is not None:
        direction = "cheaper" if econ_imp > 0 else "more expensive"
        parts.append(
            f"Economy {abs(econ_imp):.2f} RPO {direction} under pressure than their baseline."
        )
    if peco is not None:
        parts.append(f"Pressure economy: {peco:.2f}.")
    if wae is not None:
        parts.append(f"Took {wae:+.2f} wickets above model expectation under pressure.")

    return {
        "summary": f"Ranked #{rank}." if rank else "Rank not computed.",
        "highlights": parts,
        "stats": {
            "pressure_economy": peco,
            "baseline_economy": beco,
            "economy_improvement": econ_imp,
            "pressure_balls": int(pballs) if pballs else 0,
            "wickets_above_expected": wae,
        },
    }


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(pipeline_results: dict, output_dir: str) -> dict[str, str]:
    """Save ranking CSVs and analysis summary JSON.

    Parameters
    ----------
    pipeline_results : dict
        Output of ``run_full_pipeline()``.
    output_dir : str
        Directory to write output files into.

    Returns
    -------
    dict mapping logical name → absolute file path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}

    # ---- batter_rankings.csv -----------------------------------------
    batter_rankings: pd.DataFrame = pipeline_results.get("batter_rankings", pd.DataFrame())
    batter_path = out_path / "batter_rankings.csv"
    try:
        batter_rankings.to_csv(batter_path, index=False, float_format="%.4f")
        logger.info("Saved batter_rankings.csv → %s", batter_path)
        saved["batter_rankings"] = str(batter_path)
    except Exception as exc:
        logger.error("Failed to save batter_rankings.csv: %s", exc)

    # ---- bowler_rankings.csv -----------------------------------------
    bowler_rankings: pd.DataFrame = pipeline_results.get("bowler_rankings", pd.DataFrame())
    bowler_path = out_path / "bowler_rankings.csv"
    try:
        bowler_rankings.to_csv(bowler_path, index=False, float_format="%.4f")
        logger.info("Saved bowler_rankings.csv → %s", bowler_path)
        saved["bowler_rankings"] = str(bowler_path)
    except Exception as exc:
        logger.error("Failed to save bowler_rankings.csv: %s", exc)

    # ---- analysis_summary.json ---------------------------------------
    summary_path = out_path / "analysis_summary.json"
    try:
        data_summary = pipeline_results.get("data_summary", {})

        top10_batters = (
            batter_rankings["player_name"].tolist()
            if "player_name" in batter_rankings.columns
            else []
        )
        top10_bowlers = (
            bowler_rankings["player_name"].tolist()
            if "player_name" in bowler_rankings.columns
            else []
        )

        # Key batter stats
        def _extract_key_stats_bat(df: pd.DataFrame) -> dict:
            if df.empty:
                return {}
            numeric = df.select_dtypes(include=[np.number])
            return {
                "avg_adjusted_score": round(float(numeric.get("adjusted_score", pd.Series()).mean()), 3)
                    if "adjusted_score" in numeric else None,
                "avg_pressure_sr": round(float(numeric.get("pressure_strike_rate", pd.Series()).mean()), 2)
                    if "pressure_strike_rate" in numeric else None,
                "avg_sr_improvement": round(float(numeric.get("sr_improvement", pd.Series()).mean()), 2)
                    if "sr_improvement" in numeric else None,
            }

        def _extract_key_stats_bowl(df: pd.DataFrame) -> dict:
            if df.empty:
                return {}
            numeric = df.select_dtypes(include=[np.number])
            return {
                "avg_adjusted_score": round(float(numeric.get("adjusted_score", pd.Series()).mean()), 3)
                    if "adjusted_score" in numeric else None,
                "avg_pressure_economy": round(float(numeric.get("pressure_economy", pd.Series()).mean()), 2)
                    if "pressure_economy" in numeric else None,
                "avg_economy_improvement": round(float(numeric.get("economy_improvement", pd.Series()).mean()), 2)
                    if "economy_improvement" in numeric else None,
            }

        analysis_summary = {
            "data_summary": {
                k: int(v) if isinstance(v, (int, np.integer)) else v
                for k, v in data_summary.items()
            },
            "top10_batters": top10_batters,
            "top10_bowlers": top10_bowlers,
            "batter_key_stats": _extract_key_stats_bat(batter_rankings),
            "bowler_key_stats": _extract_key_stats_bowl(bowler_rankings),
        }

        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(analysis_summary, fh, indent=2, default=str)
        logger.info("Saved analysis_summary.json → %s", summary_path)
        saved["analysis_summary"] = str(summary_path)
    except Exception as exc:
        logger.error("Failed to save analysis_summary.json: %s", exc)

    return saved
