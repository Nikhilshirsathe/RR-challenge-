"""
expected_performance.py – Context-aware expected performance models for the SUPRR Selector.

Exports
-------
BattingExpectedModel
    Predicts expected runs_off_bat per legal delivery given pre-delivery context.

BowlingExpectedModel
    Predicts expected runs_charged_to_bowler per legal delivery.

WicketExpectedModel
    Predicts probability of a bowler-credited wicket per legal delivery.

fit_all_models(df, save_dir=None) -> (BattingExpectedModel, BowlingExpectedModel, WicketExpectedModel)
    Train and optionally persist all three models.

load_or_fit_models(df, save_dir) -> (BattingExpectedModel, BowlingExpectedModel, WicketExpectedModel)
    Load saved models or train fresh ones if files are absent.

add_expected_columns(df, batting_model, bowling_model, wicket_model) -> pd.DataFrame
    Append expected value and over/under columns to the ball-by-ball dataframe.

Design notes
------------
* No data leakage: all features are pre-delivery state columns.
* GroupKFold(n_splits=5) on match_id keeps entire matches together so the
  model never trains on future/same-match data.
* Models degrade gracefully for thin datasets (< MIN_TRAINING_ROWS).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, log_loss
from sklearn.model_selection import GroupKFold, cross_val_predict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streamlit cache shim
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    _cache = st.cache_data
    logger.debug("expected_performance: Streamlit detected – @st.cache_data available.")
except ImportError:
    def _cache(func=None, **_kwargs):   # type: ignore[misc]
        if func is not None:
            return func
        return lambda f: f
    logger.debug("expected_performance: Streamlit not available – using no-op cache.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_TRAINING_ROWS = 500   # below this we refuse to train and return NaN predictions

_NON_BOWLER_DISMISSALS = {
    "RunOut", "Run Out", "run out",
    "Hit Wicket", "hit wicket",
    "Obstracting the field",
    "Obstructing the field",
    "Obstructing The Field",
    "Handled the ball",
    "Retired hurt",
    "Retired out",
}

# File names for persisted models
_BATTING_FNAME  = "batting_model.joblib"
_BOWLING_FNAME  = "bowling_model.joblib"
_WICKET_FNAME   = "wicket_model.joblib"

# ---------------------------------------------------------------------------
# Phase / format encoding maps
# ---------------------------------------------------------------------------
_PHASE_MAP  = {"powerplay": 0, "middle": 1, "death": 2}
_FORMAT_MAP = {"T20": 0, "T20I": 1}

# ---------------------------------------------------------------------------
# Shared feature lists (all pre-delivery, no leakage)
# ---------------------------------------------------------------------------
_BAT_FEATURES: List[str] = [
    "pre_legal_balls_norm",        # pre_legal_balls / 120
    "pre_wickets_norm",            # pre_wickets / 10
    "pre_balls_remaining_norm",    # pre_balls_remaining / 120
    "bpi_norm",                    # bpi / 100
    "innings_phase_encoded",       # 0/1/2
    "match_format_encoded",        # 0/1
    "pre_wickets_remaining_norm",  # pre_wickets_remaining / 10
    "bpi_rrr_component_norm",      # bpi_rrr_component / 100
    "bpi_leverage_component_norm", # bpi_leverage_component / 100
    "pre_overs_complete_norm",     # pre_overs_complete / 20
]

_BOWL_FEATURES: List[str] = [
    "pre_legal_balls_norm",
    "pre_wickets_norm",
    "pre_balls_remaining_norm",
    "bopi_norm",                    # bopi / 100
    "innings_phase_encoded",
    "match_format_encoded",
    "pre_wickets_remaining_norm",
    "bopi_phase_component_norm",    # bopi_phase_component / 100
    "bopi_leverage_component_norm", # bopi_leverage_component / 100
    "pre_overs_complete_norm",
]

# Human-readable labels for feature importance exports
_BAT_FEATURE_LABELS: List[str] = [
    "pre_legal_balls", "pre_wickets", "pre_balls_remaining",
    "bpi", "innings_phase", "match_format", "pre_wickets_remaining",
    "bpi_rrr_component", "bpi_leverage_component", "pre_overs_complete",
]

_BOWL_FEATURE_LABELS: List[str] = [
    "pre_legal_balls", "pre_wickets", "pre_balls_remaining",
    "bopi", "innings_phase", "match_format", "pre_wickets_remaining",
    "bopi_phase_component", "bopi_leverage_component", "pre_overs_complete",
]

# ---------------------------------------------------------------------------
# Feature-engineering helpers
# ---------------------------------------------------------------------------

def _encode_features(df: pd.DataFrame, feature_cols: List[str],
                     label_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Return (feature_matrix, available_label_list).

    Missing source columns are filled with the column's median (train) or 0
    at predict time since we just need a valid array.
    """
    for col in feature_cols:
        if col not in df.columns:
            logger.warning("Feature column %s missing – filling with 0.", col)
            df = df.copy()
            df[col] = 0.0

    X = df[feature_cols].copy().astype(float)
    # Clip normalised values to [0, 1] to guard against out-of-range outliers
    X = X.clip(-1, 2)
    return X, label_cols


def _build_bat_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised feature columns required by the batting model."""
    out = df.copy()

    out["pre_legal_balls_norm"]        = pd.to_numeric(out.get("pre_legal_balls",        0), errors="coerce").fillna(0) / 120.0
    out["pre_wickets_norm"]            = pd.to_numeric(out.get("pre_wickets",             0), errors="coerce").fillna(0) / 10.0
    out["pre_balls_remaining_norm"]    = pd.to_numeric(out.get("pre_balls_remaining",     0), errors="coerce").fillna(0) / 120.0
    out["bpi_norm"]                    = pd.to_numeric(out.get("bpi",                    50), errors="coerce").fillna(50) / 100.0
    out["pre_wickets_remaining_norm"]  = pd.to_numeric(out.get("pre_wickets_remaining",   10), errors="coerce").fillna(10) / 10.0
    out["bpi_rrr_component_norm"]      = pd.to_numeric(out.get("bpi_rrr_component",      50), errors="coerce").fillna(50) / 100.0
    out["bpi_leverage_component_norm"] = pd.to_numeric(out.get("bpi_leverage_component", 50), errors="coerce").fillna(50) / 100.0
    out["pre_overs_complete_norm"]     = pd.to_numeric(out.get("pre_overs_complete",       0), errors="coerce").fillna(0) / 20.0

    # Phase encoding
    phase = out.get("innings_phase", pd.Series("middle", index=out.index))
    out["innings_phase_encoded"] = (
        phase.astype(str).str.lower().str.strip()
        .map(_PHASE_MAP)
        .fillna(1)
        .astype(float)
    )

    # Format encoding
    fmt = out.get("match_format", pd.Series("T20", index=out.index))
    out["match_format_encoded"] = (
        fmt.astype(str).str.upper().str.strip()
        .map(_FORMAT_MAP)
        .fillna(0)
        .astype(float)
    )

    return out


def _build_bowl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised feature columns required by the bowling / wicket models."""
    out = df.copy()

    out["pre_legal_balls_norm"]         = pd.to_numeric(out.get("pre_legal_balls",        0), errors="coerce").fillna(0) / 120.0
    out["pre_wickets_norm"]             = pd.to_numeric(out.get("pre_wickets",             0), errors="coerce").fillna(0) / 10.0
    out["pre_balls_remaining_norm"]     = pd.to_numeric(out.get("pre_balls_remaining",     0), errors="coerce").fillna(0) / 120.0
    out["bopi_norm"]                    = pd.to_numeric(out.get("bopi",                   50), errors="coerce").fillna(50) / 100.0
    out["pre_wickets_remaining_norm"]   = pd.to_numeric(out.get("pre_wickets_remaining",   10), errors="coerce").fillna(10) / 10.0
    out["bopi_phase_component_norm"]    = pd.to_numeric(out.get("bopi_phase_component",    50), errors="coerce").fillna(50) / 100.0
    out["bopi_leverage_component_norm"] = pd.to_numeric(out.get("bopi_leverage_component", 50), errors="coerce").fillna(50) / 100.0
    out["pre_overs_complete_norm"]      = pd.to_numeric(out.get("pre_overs_complete",       0), errors="coerce").fillna(0) / 20.0

    # Phase encoding
    phase = out.get("innings_phase", pd.Series("middle", index=out.index))
    out["innings_phase_encoded"] = (
        phase.astype(str).str.lower().str.strip()
        .map(_PHASE_MAP)
        .fillna(1)
        .astype(float)
    )

    # Format encoding
    fmt = out.get("match_format", pd.Series("T20", index=out.index))
    out["match_format_encoded"] = (
        fmt.astype(str).str.upper().str.strip()
        .map(_FORMAT_MAP)
        .fillna(0)
        .astype(float)
    )

    return out


def _get_bowler_credit(df: pd.DataFrame) -> pd.Series:
    """Boolean Series: True when the wicket is credited to the bowler."""
    if "dismissal_type" not in df.columns:
        logger.warning(
            "_get_bowler_credit: 'dismissal_type' column not found — "
            "crediting all is_wicket rows to the bowler."
        )
        return df["is_wicket"].astype(bool)
    dismissal     = df["dismissal_type"].fillna("").astype(str).str.strip()
    is_non_bowler = dismissal.isin(_NON_BOWLER_DISMISSALS)
    return df["is_wicket"].astype(bool) & ~is_non_bowler


# ---------------------------------------------------------------------------
# Base model mixin
# ---------------------------------------------------------------------------

class _BaseExpectedModel:
    """Shared init, save, load, and feature-importance logic."""

    _model_filename: str = "model.joblib"   # override in subclasses
    _feature_cols:   List[str] = []
    _feature_labels: List[str] = []

    def __init__(self) -> None:
        self._estimator = None          # fitted sklearn estimator
        self.metrics: Dict = {}         # CV MAE, R2, etc.
        self._is_fitted: bool = False

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. Call fit(df) first."
            )

    def get_feature_importance(self) -> Dict[str, float]:
        """Return {feature_name: importance_score} from the fitted estimator."""
        self._check_fitted()
        importances = self._estimator.feature_importances_
        labels = self._feature_labels if self._feature_labels else self._feature_cols
        return dict(zip(labels, importances.tolist()))

    def save(self, save_dir: str) -> str:
        """Persist the fitted model to *save_dir*/<filename>.  Returns path."""
        self._check_fitted()
        path = Path(save_dir) / self._model_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("%s saved → %s", self.__class__.__name__, path)
        return str(path)

    @classmethod
    def load(cls, save_dir: str) -> "_BaseExpectedModel":
        """Load a persisted model from *save_dir*/<filename>."""
        path = Path(save_dir) / cls._model_filename
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        instance = joblib.load(path)
        logger.info("%s loaded ← %s", cls.__name__, path)
        return instance


# ---------------------------------------------------------------------------
# BattingExpectedModel
# ---------------------------------------------------------------------------

class BattingExpectedModel(_BaseExpectedModel):
    """Predicts expected runs_off_bat per legal delivery.

    Model: GradientBoostingRegressor with GroupKFold CV on match_id.
    """

    _model_filename = _BATTING_FNAME
    _feature_cols   = _BAT_FEATURES
    _feature_labels = _BAT_FEATURE_LABELS

    def __init__(self) -> None:
        super().__init__()
        self._estimator = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )

    def fit(self, df: pd.DataFrame) -> "BattingExpectedModel":
        """Train on legal deliveries.  Logs CV MAE.

        Parameters
        ----------
        df : pd.DataFrame
            Enriched + pressure-annotated ball-by-ball dataframe.
        """
        logger.info("BattingExpectedModel.fit: preparing data …")

        # Filter to legal deliveries
        legal = df[df["is_legal_ball"].astype(bool)].copy()
        logger.info("  legal deliveries: %d", len(legal))

        if len(legal) < MIN_TRAINING_ROWS:
            logger.warning(
                "BattingExpectedModel.fit: only %d legal rows, need %d minimum. "
                "Model not fitted.", len(legal), MIN_TRAINING_ROWS
            )
            self.metrics = {"error": "too_few_samples", "n_rows": len(legal)}
            return self

        legal = _build_bat_features(legal)
        X, _ = _encode_features(legal, self._feature_cols, self._feature_labels)
        y    = pd.to_numeric(legal["runs_off_bat"], errors="coerce").fillna(0).values
        groups = legal["match_id"].values

        # Cross-validation (match-wise) to report CV MAE without leakage
        logger.info("BattingExpectedModel.fit: running GroupKFold(5) CV …")
        gkf = GroupKFold(n_splits=5)
        cv_preds = cross_val_predict(self._estimator, X, y, groups=groups, cv=gkf)
        cv_mae = float(mean_absolute_error(y, cv_preds))
        cv_r2  = float(r2_score(y, cv_preds))
        logger.info("  CV MAE=%.4f  R²=%.4f", cv_mae, cv_r2)

        # Final fit on all data
        logger.info("BattingExpectedModel.fit: fitting on full dataset …")
        self._estimator.fit(X, y)
        self._is_fitted = True

        self.metrics = {
            "cv_mae": cv_mae,
            "cv_r2":  cv_r2,
            "n_training_rows": len(legal),
        }
        logger.info("BattingExpectedModel.fit: complete.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return expected runs per ball.  NaN for non-legal deliveries.

        Parameters
        ----------
        df : pd.DataFrame
            Ball-by-ball dataframe (may include non-legal deliveries).

        Returns
        -------
        pd.Series
            Same index as *df*; float values for legal balls, NaN otherwise.
        """
        self._check_fitted()

        result = pd.Series(np.nan, index=df.index, dtype="float64")
        legal_mask = df["is_legal_ball"].astype(bool)
        legal = _build_bat_features(df[legal_mask].copy())

        if legal.empty:
            logger.warning("BattingExpectedModel.predict: no legal deliveries in input.")
            return result

        X, _ = _encode_features(legal, self._feature_cols, self._feature_labels)
        preds = self._estimator.predict(X).clip(0, 6)   # runs are 0-6
        result.loc[legal_mask] = preds
        return result


# ---------------------------------------------------------------------------
# BowlingExpectedModel
# ---------------------------------------------------------------------------

class BowlingExpectedModel(_BaseExpectedModel):
    """Predicts expected runs_charged_to_bowler per legal delivery."""

    _model_filename = _BOWLING_FNAME
    _feature_cols   = _BOWL_FEATURES
    _feature_labels = _BOWL_FEATURE_LABELS

    def __init__(self) -> None:
        super().__init__()
        self._estimator = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )

    def fit(self, df: pd.DataFrame) -> "BowlingExpectedModel":
        """Train on legal deliveries.  Logs CV MAE."""
        logger.info("BowlingExpectedModel.fit: preparing data …")

        legal = df[df["is_legal_ball"].astype(bool)].copy()
        logger.info("  legal deliveries: %d", len(legal))

        if len(legal) < MIN_TRAINING_ROWS:
            logger.warning(
                "BowlingExpectedModel.fit: only %d legal rows, need %d minimum. "
                "Model not fitted.", len(legal), MIN_TRAINING_ROWS
            )
            self.metrics = {"error": "too_few_samples", "n_rows": len(legal)}
            return self

        legal = _build_bowl_features(legal)
        X, _ = _encode_features(legal, self._feature_cols, self._feature_labels)
        y    = pd.to_numeric(legal["runs_charged_to_bowler"], errors="coerce").fillna(0).values
        groups = legal["match_id"].values

        logger.info("BowlingExpectedModel.fit: running GroupKFold(5) CV …")
        gkf = GroupKFold(n_splits=5)
        cv_preds = cross_val_predict(self._estimator, X, y, groups=groups, cv=gkf)
        cv_mae = float(mean_absolute_error(y, cv_preds))
        cv_r2  = float(r2_score(y, cv_preds))
        logger.info("  CV MAE=%.4f  R²=%.4f", cv_mae, cv_r2)

        logger.info("BowlingExpectedModel.fit: fitting on full dataset …")
        self._estimator.fit(X, y)
        self._is_fitted = True

        self.metrics = {
            "cv_mae": cv_mae,
            "cv_r2":  cv_r2,
            "n_training_rows": len(legal),
        }
        logger.info("BowlingExpectedModel.fit: complete.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return expected runs_charged_to_bowler per ball.  NaN for non-legal."""
        self._check_fitted()

        result = pd.Series(np.nan, index=df.index, dtype="float64")
        legal_mask = df["is_legal_ball"].astype(bool)
        legal = _build_bowl_features(df[legal_mask].copy())

        if legal.empty:
            logger.warning("BowlingExpectedModel.predict: no legal deliveries in input.")
            return result

        X, _ = _encode_features(legal, self._feature_cols, self._feature_labels)
        preds = self._estimator.predict(X).clip(0, 6)
        result.loc[legal_mask] = preds
        return result


# ---------------------------------------------------------------------------
# WicketExpectedModel
# ---------------------------------------------------------------------------

class WicketExpectedModel(_BaseExpectedModel):
    """Predicts probability of a bowler-credited wicket per legal delivery."""

    _model_filename = _WICKET_FNAME
    _feature_cols   = _BOWL_FEATURES
    _feature_labels = _BOWL_FEATURE_LABELS

    def __init__(self) -> None:
        super().__init__()
        self._estimator = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )

    def fit(self, df: pd.DataFrame) -> "WicketExpectedModel":
        """Train on legal deliveries.  Target: bowler-credited wicket (0/1)."""
        logger.info("WicketExpectedModel.fit: preparing data …")

        legal = df[df["is_legal_ball"].astype(bool)].copy()
        logger.info("  legal deliveries: %d", len(legal))

        if len(legal) < MIN_TRAINING_ROWS:
            logger.warning(
                "WicketExpectedModel.fit: only %d legal rows, need %d minimum. "
                "Model not fitted.", len(legal), MIN_TRAINING_ROWS
            )
            self.metrics = {"error": "too_few_samples", "n_rows": len(legal)}
            return self

        legal["_bowler_wicket_target"] = _get_bowler_credit(legal).astype(int)
        wicket_rate = legal["_bowler_wicket_target"].mean()
        logger.info("  bowler-wicket rate: %.3f%%", wicket_rate * 100)

        legal = _build_bowl_features(legal)
        X, _  = _encode_features(legal, self._feature_cols, self._feature_labels)
        y     = legal["_bowler_wicket_target"].values
        groups = legal["match_id"].values

        logger.info("WicketExpectedModel.fit: running GroupKFold(5) CV …")
        gkf = GroupKFold(n_splits=5)
        # predict_proba for class=1
        cv_proba = cross_val_predict(
            self._estimator, X, y, groups=groups, cv=gkf, method="predict_proba"
        )[:, 1]
        cv_logloss = float(log_loss(y, cv_proba))
        logger.info("  CV LogLoss=%.4f", cv_logloss)

        logger.info("WicketExpectedModel.fit: fitting on full dataset …")
        self._estimator.fit(X, y)
        self._is_fitted = True

        self.metrics = {
            "cv_logloss":        cv_logloss,
            "train_wicket_rate": float(wicket_rate),
            "n_training_rows":   len(legal),
        }
        logger.info("WicketExpectedModel.fit: complete.")
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Return probability of bowler wicket per delivery.  NaN for non-legal.

        Parameters
        ----------
        df : pd.DataFrame
            Ball-by-ball frame.

        Returns
        -------
        pd.Series
            Float [0, 1] for legal deliveries, NaN otherwise.
        """
        self._check_fitted()

        result = pd.Series(np.nan, index=df.index, dtype="float64")
        legal_mask = df["is_legal_ball"].astype(bool)
        legal = _build_bowl_features(df[legal_mask].copy())

        if legal.empty:
            logger.warning("WicketExpectedModel.predict_proba: no legal deliveries.")
            return result

        X, _ = _encode_features(legal, self._feature_cols, self._feature_labels)
        proba = self._estimator.predict_proba(X)[:, 1]
        result.loc[legal_mask] = proba
        return result


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def fit_all_models(
    df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> Tuple[BattingExpectedModel, BowlingExpectedModel, WicketExpectedModel]:
    """Fit all three expected-performance models.

    Parameters
    ----------
    df : pd.DataFrame
        Full enriched + pressure-annotated ball-by-ball dataframe.
    save_dir : str, optional
        Directory to persist fitted models with joblib.

    Returns
    -------
    (BattingExpectedModel, BowlingExpectedModel, WicketExpectedModel)
    """
    logger.info("fit_all_models: starting …")

    batting_model = BattingExpectedModel()
    logger.info("[1/3] Fitting BattingExpectedModel …")
    batting_model.fit(df)

    bowling_model = BowlingExpectedModel()
    logger.info("[2/3] Fitting BowlingExpectedModel …")
    bowling_model.fit(df)

    wicket_model = WicketExpectedModel()
    logger.info("[3/3] Fitting WicketExpectedModel …")
    wicket_model.fit(df)

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if batting_model._is_fitted:
            batting_model.save(save_dir)
        if bowling_model._is_fitted:
            bowling_model.save(save_dir)
        if wicket_model._is_fitted:
            wicket_model.save(save_dir)
        logger.info("fit_all_models: models saved to %s", save_dir)

    logger.info(
        "fit_all_models: done.  batting CV MAE=%s  bowling CV MAE=%s",
        batting_model.metrics.get("cv_mae", "n/a"),
        bowling_model.metrics.get("cv_mae", "n/a"),
    )
    return batting_model, bowling_model, wicket_model


def load_or_fit_models(
    df: pd.DataFrame,
    save_dir: str,
) -> Tuple[BattingExpectedModel, BowlingExpectedModel, WicketExpectedModel]:
    """Load persisted models if they exist, otherwise fit and save.

    Parameters
    ----------
    df : pd.DataFrame
        Required only when fitting (models not found on disk).
    save_dir : str
        Directory where model joblib files are (or will be) stored.

    Returns
    -------
    (BattingExpectedModel, BowlingExpectedModel, WicketExpectedModel)
    """
    save_path = Path(save_dir)

    bat_path  = save_path / _BATTING_FNAME
    bowl_path = save_path / _BOWLING_FNAME
    wkt_path  = save_path / _WICKET_FNAME

    all_exist = bat_path.exists() and bowl_path.exists() and wkt_path.exists()

    if all_exist:
        logger.info("load_or_fit_models: loading from %s …", save_dir)
        try:
            batting_model = BattingExpectedModel.load(save_dir)
            bowling_model = BowlingExpectedModel.load(save_dir)
            wicket_model  = WicketExpectedModel.load(save_dir)
            logger.info("load_or_fit_models: all models loaded successfully.")
            return batting_model, bowling_model, wicket_model
        except Exception as exc:
            logger.warning(
                "load_or_fit_models: load failed (%s) — refitting …", exc
            )

    logger.info("load_or_fit_models: model files not found or corrupt — fitting …")
    return fit_all_models(df, save_dir=save_dir)


@_cache
def add_expected_columns(
    df: pd.DataFrame,
    _batting_model: "BattingExpectedModel",
    _bowling_model: "BowlingExpectedModel",
    _wicket_model:  "WicketExpectedModel",
) -> pd.DataFrame:
    """Append expected-value and over/under-performance columns to *df*.

    Model parameters are prefixed with ``_`` so Streamlit's cache decorator
    skips hashing them (fitted sklearn estimators are not hashable).  Pass
    the same model objects each call to get cache hits on *df*.

    New columns
    -----------
    expected_runs_per_ball          float, NaN for non-legal deliveries
    expected_runs_conceded_per_ball float, NaN for non-legal deliveries
    expected_wicket_prob            float [0,1], NaN for non-legal deliveries
    runs_above_expected             runs_off_bat - expected_runs_per_ball
                                    (NaN for non-legal; batter perspective)
    runs_prevented                  expected_runs_conceded - runs_charged_to_bowler
                                    (NaN for non-legal; bowler perspective)
    wicket_above_expected           is_wicket_bowler_credit - expected_wicket_prob
                                    (NaN for non-legal)

    Parameters
    ----------
    df : pd.DataFrame
        Full enriched + pressure-annotated ball-by-ball dataframe.
    _batting_model, _bowling_model, _wicket_model
        Fitted model instances (leading underscore = not hashed by Streamlit).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with six new columns appended.
    """
    # Rebind without leading underscore for readability below
    batting_model = _batting_model
    bowling_model = _bowling_model
    wicket_model  = _wicket_model

    logger.info("add_expected_columns: generating expected-value columns …")

    out = df.copy()
    legal_mask = out["is_legal_ball"].astype(bool)

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #
    out["expected_runs_per_ball"] = (
        batting_model.predict(out)
        if batting_model._is_fitted
        else pd.Series(np.nan, index=out.index)
    )

    out["expected_runs_conceded_per_ball"] = (
        bowling_model.predict(out)
        if bowling_model._is_fitted
        else pd.Series(np.nan, index=out.index)
    )

    out["expected_wicket_prob"] = (
        wicket_model.predict_proba(out)
        if wicket_model._is_fitted
        else pd.Series(np.nan, index=out.index)
    )

    # ------------------------------------------------------------------ #
    # Batter over/under: runs_above_expected
    # ------------------------------------------------------------------ #
    runs_off_bat = pd.to_numeric(out["runs_off_bat"], errors="coerce")
    out["runs_above_expected"] = np.where(
        legal_mask,
        runs_off_bat - out["expected_runs_per_ball"],
        np.nan,
    )

    # ------------------------------------------------------------------ #
    # Bowler over/under: runs_prevented
    # ------------------------------------------------------------------ #
    runs_charged = pd.to_numeric(out["runs_charged_to_bowler"], errors="coerce")
    out["runs_prevented"] = np.where(
        legal_mask,
        out["expected_runs_conceded_per_ball"] - runs_charged,
        np.nan,
    )

    # ------------------------------------------------------------------ #
    # Bowler wicket over/under: wicket_above_expected
    # ------------------------------------------------------------------ #
    is_bowler_wicket = _get_bowler_credit(out).astype(float)
    out["wicket_above_expected"] = np.where(
        legal_mask,
        is_bowler_wicket - out["expected_wicket_prob"],
        np.nan,
    )

    logger.info(
        "add_expected_columns: complete — "
        "expected_runs mean=%.3f, expected_wicket_prob mean=%.4f",
        float(out.loc[legal_mask, "expected_runs_per_ball"].mean()) if legal_mask.any() else float("nan"),
        float(out.loc[legal_mask, "expected_wicket_prob"].mean())    if legal_mask.any() else float("nan"),
    )

    return out
