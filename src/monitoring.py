"""Monitor changes in product price distributions."""

from pathlib import Path
import json

import pandas as pd


def main():
    df = pd.read_csv("data/processed/modelling_dataset.csv")

    train = df[df["CatEdition"].between(88, 92)]
    future = df[df["CatEdition"] == 93]

    train_median = float(train["SalePriceIncVAT"].median())
    future_median = float(future["SalePriceIncVAT"].median())

    change_pct = (
        (future_median - train_median) / train_median * 100
        if train_median != 0 else None
    )

    report = {
        "feature": "SalePriceIncVAT",
        "training_median": train_median,
        "future_median": future_median,
        "percentage_change": change_pct,
        "drift_alert": (
            abs(change_pct) > 20
            if change_pct is not None else None
        ),
        "alert_threshold_percent": 20
    }

    Path("outputs/tables").mkdir(parents=True, exist_ok=True)

    with open("outputs/tables/monitoring_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Model monitoring report:")
    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nMonitoring report saved successfully!")


if __name__ == "__main__":
    main()
