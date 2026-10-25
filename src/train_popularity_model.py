"""
Train a popularity regression model for TuneWise AI.

Run from the project root:

    python src/train_popularity_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Allow running as `python src/train_popularity_model.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (  # noqa: E402
    INDIAN_SONGS_PATH,
    LANGUAGE_COLUMN,
    MODELS_DIR,
    NUMERIC_AUDIO_FEATURES,
    POPULARITY_METRICS_PATH,
    POPULARITY_MODEL_PATH,
    REGRESSION_NUMERIC_FEATURES,
    TARGET_COLUMN,
    DatasetError,
    clean_indian_songs,
    load_indian_songs,
    prepare_regression_frame,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2


def _to_project_relative(path: Path) -> str:
    """Store portable relative paths in saved metadata when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build preprocessing that one-hot encodes language and passes numerics through."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older scikit-learn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", encoder, categorical_features),
        ],
        remainder="drop",
    )


def build_regressor_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Create the scikit-learn Pipeline used for popularity regression."""
    nums = numeric_features or REGRESSION_NUMERIC_FEATURES
    cats = categorical_features or [LANGUAGE_COLUMN]

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(nums, cats)),
            ("model", model),
        ]
    )


def train_popularity_model(
    csv_path: Path | str | None = None,
    *,
    model_path: Path | str | None = None,
    metrics_path: Path | str | None = None,
) -> dict:
    """
    Train and persist the popularity regressor.

    Returns a dictionary with metrics and artifact paths.
    """
    source = Path(csv_path) if csv_path else INDIAN_SONGS_PATH
    out_model = Path(model_path) if model_path else POPULARITY_MODEL_PATH
    out_metrics = Path(metrics_path) if metrics_path else POPULARITY_METRICS_PATH

    print("=" * 60)
    print("TuneWise AI — Popularity Regression Training")
    print("=" * 60)
    print(f"Dataset path: {source}")

    raw = load_indian_songs(source)
    print(f"Raw rows loaded: {len(raw):,}")

    cleaned = clean_indian_songs(raw)
    print(f"Rows after cleaning / de-duplication: {len(cleaned):,}")
    print(f"Languages: {sorted(cleaned[LANGUAGE_COLUMN].unique().tolist())}")

    frame = prepare_regression_frame(cleaned)
    feature_cols = REGRESSION_NUMERIC_FEATURES + [LANGUAGE_COLUMN]
    X = frame[feature_cols]
    y = frame[TARGET_COLUMN]
    artists = frame["artist_name"] if "artist_name" in frame.columns else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")

    # Baseline: median DummyRegressor
    baseline = Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(REGRESSION_NUMERIC_FEATURES, [LANGUAGE_COLUMN]),
            ),
            ("model", DummyRegressor(strategy="median")),
        ]
    )
    print("\nFitting median DummyRegressor baseline...")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_regression(y_test, baseline_pred)
    print(
        "Baseline — "
        f"MAE={baseline_metrics['mae']:.3f}, "
        f"RMSE={baseline_metrics['rmse']:.3f}, "
        f"R2={baseline_metrics['r2']:.4f}"
    )

    pipeline = build_regressor_pipeline()
    print("\nFitting RandomForestRegressor pipeline...")
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    # Clip predictions to Spotify popularity scale for reporting
    pred_clipped = np.clip(pred, 0, 100)
    model_metrics = evaluate_regression(y_test, pred_clipped)
    print(
        "RandomForest — "
        f"MAE={model_metrics['mae']:.3f}, "
        f"RMSE={model_metrics['rmse']:.3f}, "
        f"R2={model_metrics['r2']:.4f}"
    )

    group_metrics = None
    if artists is not None and artists.nunique() > 5:
        print("\nRunning artist-grouped holdout (stronger leakage check)...")
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=artists))
        group_pipe = build_regressor_pipeline()
        group_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        group_pred = np.clip(group_pipe.predict(X.iloc[test_idx]), 0, 100)
        group_metrics = evaluate_regression(y.iloc[test_idx], group_pred)
        print(
            "Artist-grouped holdout — "
            f"MAE={group_metrics['mae']:.3f}, "
            f"RMSE={group_metrics['rmse']:.3f}, "
            f"R2={group_metrics['r2']:.4f}"
        )
        print(
            "Note: random splits can look optimistic when songs by the same "
            "artist appear in both train and test sets."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "pipeline": pipeline,
        "feature_names": feature_cols,
        "numeric_features": REGRESSION_NUMERIC_FEATURES,
        "categorical_features": [LANGUAGE_COLUMN],
        "audio_features": NUMERIC_AUDIO_FEATURES,
        "target": TARGET_COLUMN,
        "languages_seen": sorted(cleaned[LANGUAGE_COLUMN].unique().tolist()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "metrics": {
            "random_split": model_metrics,
            "baseline_median": baseline_metrics,
            "artist_group_holdout": group_metrics,
        },
        "dataset_path": _to_project_relative(source),
        "model_type": "RandomForestRegressor",
        "notes": [
            "Predictions are model estimates on a 0-100 popularity scale.",
            "The model does not guarantee future Spotify popularity.",
            "Many songs have near-zero popularity, which affects metrics.",
            "Random train/test splits may overestimate performance when "
            "tracks by the same artist appear in both sets.",
        ],
    }

    joblib.dump(payload, out_model)
    print(f"\nSaved model -> {out_model}")

    metrics_payload = {
        "dataset_rows_raw": int(len(raw)),
        "dataset_rows_clean": int(len(cleaned)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "languages": {
            str(k): int(v)
            for k, v in cleaned[LANGUAGE_COLUMN].value_counts().items()
        },
        "baseline_median": baseline_metrics,
        "random_forest_random_split": model_metrics,
        "random_forest_artist_group_holdout": group_metrics,
        "model_path": _to_project_relative(out_model),
        "feature_names": feature_cols,
    }
    out_metrics.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics -> {out_metrics}")
    print("=" * 60)

    return {
        "model_path": out_model,
        "metrics_path": out_metrics,
        "metrics": metrics_payload,
        "payload": payload,
    }


def main() -> int:
    try:
        train_popularity_model()
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected training failure: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
