"""Explore classification thresholds using historical validation data."""

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score, f1_score

from src.feature_engineering import engineer_features
from src.feature_selection import RETAINED_FEATURES
from src.preprocessing import build_preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def main():
    df = pd.read_csv("data/processed/modelling_dataset.csv")
    df = engineer_features(df)

    train = df[df["CatEdition"].between(88, 91)]
    validation = df[df["CatEdition"] == 92]

    X_train = train[RETAINED_FEATURES]
    y_train = train["DiscontinuedTF"].astype(int)

    X_val = validation[RETAINED_FEATURES]
    y_val = validation["DiscontinuedTF"].astype(int)

    model = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42
        )),
    ])

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_val)[:, 1]

    results = []

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        predictions = (probabilities >= threshold).astype(int)

        results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "flagged_percent": predictions.mean() * 100,
        })

    results = pd.DataFrame(results)
    results.to_csv(
        "outputs/tables/threshold_validation.csv",
        index=False
    )

    print("\nThreshold analysis on validation edition 92:")
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
