"""Assess missing values, duplicates and feature types."""

from src.data_ingestion import load_datasets


def analyse_quality(df, name):
    """Print reproducible data-quality diagnostics."""
    print(f"\n{'=' * 60}")
    print(f"DATA QUALITY: {name}")
    print(f"{'=' * 60}")

    print("\nMissing values (%):")
    print(
        df.isna().mean().mul(100).round(2)
        .sort_values(ascending=False).to_string()
    )

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nFeature data types:")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    products, catalogue, dictionary = load_datasets()

    analyse_quality(products, "ProductDetails")
    analyse_quality(catalogue, "CatalogueDiscontinuation")
    analyse_quality(dictionary, "DataDictionary")
