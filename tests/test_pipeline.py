"""Automated tests for the Retalix ML pipeline."""

import numpy as np
import pandas as pd

from src.feature_engineering import engineer_features
from src.feature_selection import RETAINED_FEATURES
from src.preprocessing import build_preprocessor
from src.train_test_split import get_train_test_data


def test_feature_engineering():
    sample = pd.DataFrame({
        "ForecastPerWeek": [100.0, 50.0],
        "ActualsPerWeek": [80.0, 70.0],
    })

    result = engineer_features(sample)

    assert result["ForecastActualGap"].tolist() == [20.0, -20.0]


def test_train_test_shapes():
    X_train, X_test, y_train, y_test = get_train_test_data()

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == len(RETAINED_FEATURES)
    assert X_test.shape[1] == len(RETAINED_FEATURES)


def test_preprocessing_no_missing_values():
    X_train, _, _, _ = get_train_test_data()

    # Small sample keeps this automated test fast.
    sample = X_train.head(100)

    transformed = build_preprocessor().fit_transform(sample)

    assert transformed.shape[0] == 100

    values = (
        transformed.data
        if hasattr(transformed, "toarray")
        else np.asarray(transformed)
    )

    assert np.isfinite(values).all()


def test_no_target_leakage():
    assert "DiscontinuedTF" not in RETAINED_FEATURES
    assert "Status" not in RETAINED_FEATURES
    assert "ProductKey" not in RETAINED_FEATURES
