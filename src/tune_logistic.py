"""Tune Logistic Regression using chronological cross-validation."""

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.chronological_cv import get_chronological_folds
from src.feature_engineering import engineer_features
from src.feature_selection import RETAINED_FEATURES
from src.preprocessing import build_preprocessor


def main():
    df = pd.read_csv("data/processed/modelling_dataset.csv")

    # Edition 93 is reserved for final evaluation.
    train = df[df["CatEdition"].between(88, 92)].reset_index(drop=True)
    train = engineer_features(train)

    X = train[RETAINED_FEATURES]
    y = train["DiscontinuedTF"].astype(int)

    pipeline = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        )),
    ])

    parameters = {
        "classifier__C": [0.01, 0.1, 1.0],
        "classifier__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring="average_precision",
        cv=get_chronological_folds(),
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(X, y)

    Path("models").mkdir(exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(search.cv_results_)
    results.to_csv(
        "outputs/tables/logistic_grid_search.csv",
        index=False
    )

    joblib.dump(
        search.best_estimator_,
        "models/logistic_regression_tuned_v1.joblib"
    )

    summary = {
        "best_parameters": search.best_params_,
        "best_cv_average_precision": float(search.best_score_),
        "number_of_candidates": len(results),
        "number_of_folds": 3,
    }

    with open(
        "outputs/tables/logistic_tuning_summary.json", "w"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\nBest parameters:", search.best_params_)
    print("Best CV average precision:", round(search.best_score_, 4))
    print("Parameter combinations:", len(results))
    print("Tuned model and experiment results saved!")


if __name__ == "__main__":
    main()
