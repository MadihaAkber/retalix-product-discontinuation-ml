"""Tune Random Forest using chronological cross-validation."""

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.chronological_cv import get_chronological_folds
from src.feature_engineering import engineer_features
from src.feature_selection import RETAINED_FEATURES
from src.preprocessing import build_preprocessor


def main():
    df = pd.read_csv("data/processed/modelling_dataset.csv")

    train = df[df["CatEdition"].between(88, 92)].reset_index(drop=True)
    train = engineer_features(train)

    X = train[RETAINED_FEATURES]
    y = train["DiscontinuedTF"].astype(int)

    pipeline = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            random_state=42,
            n_jobs=1,
        )),
    ])

    parameters = {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_leaf": [5, 10],
        "classifier__max_features": ["sqrt", 0.5],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=6,
        scoring="average_precision",
        cv=get_chronological_folds(),
        random_state=42,
        n_jobs=2,
        refit=True,
        verbose=1,
    )

    search.fit(X, y)

    Path("models").mkdir(exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(search.cv_results_)
    results.to_csv(
        "outputs/tables/random_forest_search.csv",
        index=False
    )

    joblib.dump(
        search.best_estimator_,
        "models/random_forest_tuned_v1.joblib"
    )

    summary = {
        "best_parameters": search.best_params_,
        "best_cv_average_precision": float(search.best_score_),
        "number_of_candidates": len(results),
        "number_of_folds": 3,
    }

    with open(
        "outputs/tables/random_forest_tuning_summary.json", "w"
    ) as f:
        json.dump(summary, f, indent=4)

    print("\nBest parameters:", search.best_params_)
    print("Best CV average precision:", round(search.best_score_, 4))
    print("Parameter combinations:", len(results))
    print("Tuned model and experiment results saved!")


if __name__ == "__main__":
    main()
