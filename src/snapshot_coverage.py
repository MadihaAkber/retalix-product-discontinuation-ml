"""Investigate representativeness of Week -24 snapshots."""

from pathlib import Path

import pandas as pd

from src.data_ingestion import load_datasets


def main():
    _, catalogue, _ = load_datasets()

    catalogue = catalogue[
        catalogue["CatEdition"].between(88, 93)
    ].copy()

    # One record per product per catalogue edition.
    products = catalogue.groupby(
        ["CatEdition", "ProductKey"], as_index=False
    ).agg(
        DiscontinuedTF=("DiscontinuedTF", "first"),
        earliest_week=("WeeksOut", "min")
    )

    products["has_week_24"] = products["earliest_week"].eq(-24)

    summary = products.groupby("has_week_24").agg(
        products=("ProductKey", "size"),
        discontinuation_rate=("DiscontinuedTF", "mean")
    )

    summary["discontinuation_rate"] *= 100

    print("\nSnapshot coverage comparison:")
    print(summary.round(2).to_string())

    print("\nCoverage by catalogue edition:")
    coverage = products.groupby("CatEdition")["has_week_24"].agg(
        ["count", "sum", "mean"]
    )
    coverage["mean"] *= 100
    print(coverage.round(2).to_string())

    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv("outputs/tables/snapshot_coverage_summary.csv")
    coverage.to_csv("outputs/tables/snapshot_coverage_by_edition.csv")

    print("\nSnapshot coverage analysis saved successfully!")


if __name__ == "__main__":
    main()
