"""Create domain-informed retail prediction features."""

import pandas as pd


def engineer_features(df):
    """Create features using information available at prediction time."""
    result = df.copy()

    # Difference between expected and observed weekly demand.
    result["ForecastActualGap"] = (
        result["ForecastPerWeek"] - result["ActualsPerWeek"]
    )

    return result


if __name__ == "__main__":
    df = pd.read_csv("data/processed/modelling_dataset.csv")
    engineered = engineer_features(df)

    print("Original columns:", len(df.columns))
    print("Engineered columns:", len(engineered.columns))
    print("\nNew feature statistics:")
    print(engineered["ForecastActualGap"].describe())

    print("\nFirst five examples:")
    print(
        engineered[
            ["ForecastPerWeek", "ActualsPerWeek", "ForecastActualGap"]
        ].head().to_string(index=False)
    )
