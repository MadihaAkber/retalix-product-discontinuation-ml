"""Exploratory analysis of Retalix product discontinuation."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_ingestion import load_datasets


def main():
    _, catalogue, _ = load_datasets()

    # One target observation per product per catalogue edition.
    product_level = (
        catalogue.groupby(["CatEdition", "ProductKey"], as_index=False)
        ["DiscontinuedTF"].first()
    )

    # Exclude incomplete, single-class edition 94.
    product_level = product_level[
        product_level["CatEdition"].between(88, 93)
    ].copy()

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.countplot(
        data=product_level,
        x="CatEdition",
        hue="DiscontinuedTF",
        palette="Set2"
    )

    plt.title("Product Discontinuation by Catalogue Edition")
    plt.xlabel("Catalogue Edition")
    plt.ylabel("Number of Products")
    plt.legend(title="Discontinued")
    plt.tight_layout()

    plt.savefig(
        "outputs/figures/discontinuation_by_edition.png",
        dpi=300
    )
    plt.close()

    print("Graph saved successfully!")
    print("outputs/figures/discontinuation_by_edition.png")


if __name__ == "__main__":
    main()
