"""Train and evaluate a baseline Logistic Regression model."""

from pathlib import Path
import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.preprocessing import build_preprocessor
from src.train_test_split import get_train_test_data


def main():
    X_train, X_test, y_train, y_test = get_train_test_data()

    model = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
    }

    Path("models").mkdir(exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)

    joblib.dump(model, "models/logistic_regression_v1.joblib")

    with open("outputs/tables/logistic_baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Logistic Regression baseline results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nModel and metrics saved successfully!")


if __name__ == "__main__":
    main()
