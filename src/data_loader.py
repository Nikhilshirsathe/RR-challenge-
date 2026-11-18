"""
data_loader.py - Load and merge all datasets into a unified ball-by-ball dataframe.

Adds: match_format, season, comp_name, player_name lookups, is_legal_ball, is_wide,
      is_noball, runs_off_bat, runs_charged_to_bowler, total_runs_on_ball, is_super_over.
Uses Streamlit caching.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Try to import streamlit cache; fall back to a no-op decorator when running
# outside a Streamlit session (e.g. in tests or direct Python execution).
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    _cache = st.cache_data
except ImportError:
    def _cache(*args, **kwargs):  # type: ignore[misc]
        """No-op replacement for st.cache_data when Streamlit is unavailable."""
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\Nikhi\Downloads\6af458274c1e8_data_pack\Data Pack")

_FILE_MAP = {
    "ball_by_ball":                   DATA_DIR / "ball_by_ball.csv",
    "matches":                        DATA_DIR / "matches.csv",
    "playing_eleven":                 DATA_DIR / "playing_eleven.csv",
    "competition_team_player_mapping": DATA_DIR / "competition_team_player_mapping.csv",
    "competitions":                   DATA_DIR / "competitions.csv",
    "teams":                          DATA_DIR / "teams.csv",
}

# ---------------------------------------------------------------------------
# Extras classification constants
# ---------------------------------------------------------------------------
_WIDE      = "wide"
_NOBALL    = "Overstep"
_LEGBYES   = "legbyes"
_BYES      = "byes"
_EXTRAS_LEGAL_BALL_EXCLUDES = {_WIDE}   # wides are the only non-legal balls


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@_cache(show_spinner="Loading raw data files…")
def load_all_data() -> dict[str, pd.DataFrame]:
    """Load every CSV into a dict keyed by logical name.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ball_by_ball, matches, playing_eleven,
              competition_team_player_mapping, competitions, teams.
    """
    result: dict[str, pd.DataFrame] = {}
    for name, path in _FILE_MAP.items():
        if not path.exists():
            logger.error("Missing file: %s", path)
            raise FileNotFoundError(f"Required data file not found: {path}")
        logger.info("Reading %s (%s)", name, path.name)
        result[name] = pd.read_csv(path, low_memory=False)
        logger.info("  → %d rows × %d cols", *result[name].shape)
    return result


@_cache(show_spinner="Building enriched ball-by-ball dataset…")
def load_enriched_bbb() -> pd.DataFrame:
    """Return ball-by-ball data merged with match / competition context plus
    all derived analytical columns.

    Derived columns added
    ---------------------
    is_legal_ball           – True when extras_type is NOT 'wide'
    is_wide                 – extras_type == 'wide'
    is_noball               – extras_type == 'Overstep'
    is_bye                  – extras_type in ('byes', 'legbyes')
    runs_off_bat            – the `runs` column (batter runs)
    runs_charged_to_bowler  – see extras rules in module docstring
    total_runs_on_ball      – runs + extras1_runs + extras_2_runs
    is_boundary_4           – alias for is_four
    is_boundary_6           – alias for is_six
    is_super_over           – innings >= 3
    innings_phase           – 'powerplay' / 'middle' / 'death'
    match_format            – from competitions
    season                  – from competitions
    comp_name               – from competitions
    """
    data = load_all_data()
    bbb   = data["ball_by_ball"].copy()
    matches = data["matches"]
    comps   = data["competitions"]

    # ------------------------------------------------------------------
    # 1. Merge match metadata
    # ------------------------------------------------------------------
    bbb = bbb.merge(
        matches[["match_id", "comp_id",
                 "first_inn_revised_overs", "second_inn_revised_overs",
                 "revised_target"]],
        on="match_id",
        how="left",
    )
    logger.info("After match merge: %d rows", len(bbb))

    # ------------------------------------------------------------------
    # 2. Merge competition metadata
    # ------------------------------------------------------------------
    bbb = bbb.merge(
        comps[["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id",
        how="left",
    )
    logger.info("After competition merge: %d rows", len(bbb))

    # ------------------------------------------------------------------
    # 3. Normalise numeric columns
    # ------------------------------------------------------------------
    for col in ("extras1_runs", "extras_2_runs", "runs"):
        bbb[col] = pd.to_numeric(bbb[col], errors="coerce").fillna(0)

    # ------------------------------------------------------------------
    # 4. extras_type as string — keep NaN as the string "NaN" won't match
    #    comparisons, so we use pd.isna() checks inside the formula.
    # ------------------------------------------------------------------
    extras = bbb["extras_type"].astype(str)   # NaN → 'nan'

    # ------------------------------------------------------------------
    # 5. Delivery-classification flags
    # ------------------------------------------------------------------
    bbb["is_wide"]    = extras == _WIDE
    bbb["is_noball"]  = extras == _NOBALL
    bbb["is_bye"]     = extras.isin([_BYES, _LEGBYES])
    bbb["is_legal_ball"] = ~bbb["is_wide"]

    # ------------------------------------------------------------------
    # 6. Run columns
    # ------------------------------------------------------------------
    bbb["runs_off_bat"]       = bbb["runs"].astype(float)
    bbb["total_runs_on_ball"] = (
        bbb["runs"].astype(float)
        + bbb["extras1_runs"].astype(float)
        + bbb["extras_2_runs"].astype(float)
    )

    # runs_charged_to_bowler:
    #   wide      → extras1_runs (wide penalty; runs_off_bat is always 0)
    #   Overstep  → runs_off_bat + extras1_runs (1 no-ball penalty + batter runs)
    #   byes/legbyes → 0 extra charged (runs_off_bat typically 0 too)
    #   normal/other → runs_off_bat
    conditions = [
        bbb["is_wide"],
        bbb["is_noball"],
        bbb["is_bye"],
    ]
    choices = [
        bbb["extras1_runs"].astype(float),
        bbb["runs_off_bat"] + bbb["extras1_runs"].astype(float),
        bbb["runs_off_bat"],                   # byes/legbyes: no extras charged
    ]
    bbb["runs_charged_to_bowler"] = np.select(
        conditions, choices, default=bbb["runs_off_bat"]
    )

    # ------------------------------------------------------------------
    # 7. Boundary / super-over flags
    # ------------------------------------------------------------------
    bbb["is_boundary_4"] = bbb["is_four"].astype(bool)
    bbb["is_boundary_6"] = bbb["is_six"].astype(bool)
    bbb["is_super_over"] = bbb["innings"].astype(int) >= 3

    # ------------------------------------------------------------------
    # 8. Innings phase
    # ------------------------------------------------------------------
    bbb["innings_phase"] = pd.cut(
        bbb["overid"].astype(int),
        bins=[0, 6, 15, 20],
        labels=["powerplay", "middle", "death"],
        right=True,
    ).astype(str)

    logger.info("Enriched BBB built: %d rows, %d cols", *bbb.shape)
    return bbb


@_cache(show_spinner=False)
def get_player_lookup() -> dict[int, str]:
    """Return {player_id: player_name} built from competition_team_player_mapping.

    When the same player_id appears multiple times we take the last (most
    recent by row order, which preserves latest comp data).
    """
    data = load_all_data()
    mapping = data["competition_team_player_mapping"]

    # Deduplicate: keep last occurrence of each player_id
    deduped = (
        mapping[["player_id", "player_name"]]
        .dropna(subset=["player_id"])
        .drop_duplicates(subset=["player_id"], keep="last")
    )
    lookup: dict[int, str] = dict(
        zip(deduped["player_id"].astype(int), deduped["player_name"].astype(str))
    )
    logger.info("Player lookup built: %d entries", len(lookup))
    return lookup


@_cache(show_spinner=False)
def get_team_lookup() -> dict[tuple[int, int], str]:
    """Return {(comp_id, team_id): team_name}.

    For cases where a team_id appears in multiple competitions, the latest
    comp (highest comp_id) entry is used for the global team_id → name
    mapping as a convenience; the full tuple key preserves per-comp accuracy.
    """
    data = load_all_data()
    teams = data["teams"]
    comps = data["competitions"]

    # Attach season so we can sort by recency
    teams_with_season = teams.merge(
        comps[["comp_id", "season"]],
        on="comp_id",
        how="left",
    ).sort_values("season", ascending=True)   # ascending so last = latest

    lookup: dict[tuple[int, int], str] = {}
    for _, row in teams_with_season.iterrows():
        lookup[(int(row["comp_id"]), int(row["team_id"]))] = str(row["team_name"])

    logger.info("Team lookup built: %d entries", len(lookup))
    return lookup
