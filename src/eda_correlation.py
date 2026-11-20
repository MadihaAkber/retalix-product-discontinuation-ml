"""Explore correlations between numeric retail features."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_ingestion import load_datasets


def main():
    _, catalogue, _ = load_datasets()

    # Use observations from editions 88–93.
    data = catalogue[
        catalogue["CatEdition"].between(88, 93)
    ].copy()

    # Exclude ProductKey because it is an identifier.
    columns = [
        "WeeksOut",
        "SalePriceIncVAT",
        "ForecastPerWeek",
        "ActualsPerWeek",
    ]

    correlation = data[columns].corr(method="spearman")

    print("\nSpearman Correlation Matrix:")
    print(correlation.round(3).to_string())

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True
    )

    plt.title("Spearman Correlation Between Retail Features")
    plt.tight_layout()
    plt.savefig(
        "outputs/figures/correlation_heatmap.png",
        dpi=300
    )
    plt.close()

    print("\nCorrelation heatmap saved successfully!")


if __name__ == "__main__":
    main()
