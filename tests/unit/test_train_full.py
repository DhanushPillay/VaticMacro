"""Train integration — cover train() without heavy compute via mocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class _DummyGrid:
    def __init__(self, pipe, param_grid, cv=None, scoring=None, n_jobs=None):
        self.pipe = pipe
        self.param_grid = param_grid
        self.best_estimator_ = None
        self.best_params_ = {}

    def fit(self, X, y):
        try:
            # try to fit real pipe quickly
            self.best_estimator_ = self.pipe
            self.best_estimator_.fit(X, y)
        except Exception:
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import RobustScaler
            from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression

            k = 5 if X.shape[1] >= 5 else X.shape[1]
            # ensure k not > n_features
            simple = Pipeline(
                [
                    ("variance", VarianceThreshold(threshold=0.01)),
                    ("select", SelectKBest(f_regression, k=k)),
                    ("scaler", RobustScaler()),
                    ("model", Ridge()),
                ]
            )
            simple.fit(X, y)
            self.best_estimator_ = simple
        # populate best_params from param_grid first values
        bp = {}
        for kk, vv in self.param_grid.items():
            if isinstance(vv, list) and len(vv) > 0:
                bp[kk] = vv[0]
            else:
                bp[kk] = vv
        # ensure required keys
        bp.setdefault("model__alpha", 1.0)
        bp.setdefault("select__k", 10)
        bp.setdefault("model__l1_ratio", 0.5)
        bp.setdefault("model__n_estimators", 100)
        bp.setdefault("model__max_depth", 3)
        bp.setdefault("model__min_samples_leaf", 10)
        bp.setdefault("model__learning_rate", 0.05)
        bp.setdefault("model__num_leaves", 15)
        self.best_params_ = bp
        return self


def _fake_cross_validate(estimator, X, y, cv=None, scoring=None, return_train_score=False):
    return {
        "test_r2": np.array([0.5, 0.55, 0.6, 0.52, 0.58]),
        "test_neg_mean_absolute_error": np.array([-0.9, -0.85, -0.8, -0.88, -0.82]),
        "test_neg_root_mean_squared_error": np.array([-1.1, -1.05, -1.0, -1.08, -1.02]),
    }


def test_train_full_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.train_model as tm

    from src.feature_engineering import create_features

    df_raw = pd.read_csv("data/inflation_dataset.csv")
    feat = create_features(df_raw)
    # feat has Date, CPI, current_inflation_yoy + 29 others = 293 rows
    assert len(feat) > 50
    assert "current_inflation_yoy" in feat.columns

    # chdir to tmp so models/* writes are isolated
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)

    # patch heavy sklearn parts
    monkeypatch.setattr(tm, "GridSearchCV", _DummyGrid)
    monkeypatch.setattr(tm, "cross_validate", _fake_cross_validate)
    # also patch sklearn.model_selection if train imports there? already patched via tm

    result = tm.train(feat)
    assert result is not None

    assert (tmp_path / "models" / "best_model.pkl").exists()
    assert (tmp_path / "models" / "metrics.json").exists()
    assert (tmp_path / "models" / "holdout.csv").exists()

    metrics = json.loads((tmp_path / "models" / "metrics.json").read_text())
    assert metrics["best_model"] in [
        "Ridge",
        "Ridge(MI)",
        "ElasticNet",
        "Ridge(Poly)",
        "RandomForest",
        "XGBoost",
        "LightGBM",
    ]
    assert len(metrics["metrics"]) == 7
    for m in metrics["metrics"]:
        assert "r2_mean" in m
        assert "mae" in m

    hold = pd.read_csv(tmp_path / "models" / "holdout.csv")
    assert "Predicted_Inflation" in hold.columns
    assert "target_future_inflation" in hold.columns

    # check artifact contains environment
    import joblib

    art = joblib.load(tmp_path / "models" / "best_model.pkl")
    assert "pipeline" in art
    assert "feature_columns" in art
    assert "environment" in art
    assert "scikit-learn" in art["environment"]


def test_train_handles_single_row_nan():
    from src.train_model import compute_metrics, naive_baseline_mae

    # edge: single value
    m = compute_metrics([1.0], [1.0])
    assert m["r2"] == 0.0
    assert np.isnan(naive_baseline_mae([1.0]))
