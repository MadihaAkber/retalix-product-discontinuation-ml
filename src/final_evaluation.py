"""Evaluate the selected model on held-out catalogue edition 93."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.train_test_split import get_train_test_data


def main():
    _, X_test, _, y_test = get_train_test_data()

    model = joblib.load(
        "models/logistic_regression_tuned_v1.joblib"
    )

    predictions = (model.predict_proba(X_test)[:, 1] >= 0.10).astype(int)
    probabilities = model.predict_proba(X_test)[:, 1]

    figures = Path("outputs/figures")
    figures.mkdir(parents=True, exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Retained", "Discontinued"],
        cmap="Blues",
        values_format="d",
    )
    plt.title("Logistic Regression: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures / "final_confusion_matrix.png", dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("Logistic Regression: ROC Curve")
    plt.tight_layout()
    plt.savefig(figures / "final_roc_curve.png", dpi=200)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, probabilities)
    plt.title("Logistic Regression: Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(figures / "final_precision_recall_curve.png", dpi=200)
    plt.close()

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
    }

    with open(
        "outputs/tables/final_evaluation_metrics.json", "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    print("Final evaluation on edition 93:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nThree evaluation graphs saved successfully!")


if __name__ == "__main__":
    main()
