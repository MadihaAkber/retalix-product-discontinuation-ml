"""Assess whether Week -24 availability differs across product groups."""

from pathlib import Path
import pandas as pd

from src.data_ingestion import load_datasets


def main():
    products, catalogue, _ = load_datasets()

    catalogue = catalogue[
        catalogue["CatEdition"].between(88, 93)
    ]

    product_editions = (
        catalogue.groupby(
            ["CatEdition", "ProductKey"], as_index=False
        )
        .agg(first_week=("WeeksOut", "min"))
    )

    product_editions["has_week_24"] = (
        product_editions["first_week"] == -24
    )

    data = product_editions.merge(
        products,
        on="ProductKey",
        how="left",
        validate="many_to_one"
    )

    for feature in ["DIorDOM", "Seasonal"]:
        summary = (
            data.groupby(feature, dropna=False)["has_week_24"]
            .agg(total="size", available="sum", coverage="mean")
            .reset_index()
        )

        summary["coverage"] = (
            summary["coverage"] * 100
        ).round(2)

        print(f"\nSnapshot coverage by {feature}:")
        print(summary.to_string(index=False))

        Path("outputs/tables").mkdir(
            parents=True, exist_ok=True
        )
        summary.to_csv(
            f"outputs/tables/snapshot_bias_{feature}.csv",
            index=False
        )


if __name__ == "__main__":
    main()
