"""Evaluate the validation-selected threshold on the final test set."""

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.train_test_split import get_train_test_data


def main():
    _, X_test, _, y_test = get_train_test_data()

    model = joblib.load(
        "models/logistic_regression_tuned_v1.joblib"
    )

    threshold = 0.1
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "flagged_percent": float(predictions.mean() * 100),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    output = Path("outputs/tables/threshold_final_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=4))

    print("Final test results at threshold 0.1:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nResults saved successfully!")


if __name__ == "__main__":
    main()
