"""Compare discontinuation rates by product sourcing type."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_ingestion import load_datasets


def main():
    products, catalogue, _ = load_datasets()

    # Use one record per product and edition.
    product_level = (
        catalogue[catalogue["CatEdition"].between(88, 93)]
        .groupby(["CatEdition", "ProductKey"], as_index=False)
        ["DiscontinuedTF"].first()
    )

    merged = product_level.merge(
        products[["ProductKey", "DIorDOM"]],
        on="ProductKey",
        how="left",
        validate="many_to_one"
    )

    rates = (
        merged.groupby("DIorDOM")["DiscontinuedTF"]
        .mean()
        .mul(100)
        .reset_index(name="DiscontinuationRate")
    )

    print("\nDiscontinuation rates (%):")
    print(rates.to_string(index=False))

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))

    ax = sns.barplot(
        data=rates,
        x="DIorDOM",
        y="DiscontinuationRate",
        hue="DIorDOM",
        palette="Set2",
        legend=False
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)

    plt.title("Discontinuation Rate by Product Sourcing Type")
    plt.xlabel("Product Sourcing Type")
    plt.ylabel("Discontinuation Rate (%)")
    plt.ylim(0, 100)
    plt.tight_layout()

    plt.savefig(
        "outputs/figures/discontinuation_by_sourcing.png",
        dpi=300
    )
    plt.close()

    print("\nSecond graph saved successfully!")


if __name__ == "__main__":
    main()
