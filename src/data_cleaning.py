"""Validate and clean Retalix catalogue records."""

from src.data_ingestion import load_datasets


def clean_catalogue(df):
    """Apply justified consistency checks and fixes."""
    cleaned = df.copy()

    # Standardise text categories to avoid inconsistent labels.
    cleaned["Status"] = (
        cleaned["Status"].astype("string").str.strip().str.upper()
    )

    # Validate permitted status categories.
    assert cleaned["Status"].isin(["RI", "RO"]).all(), \
        "Unexpected status category detected"

    # Validate non-negative numeric measures.
    numeric_cols = [
        "SalePriceIncVAT",
        "ForecastPerWeek",
        "ActualsPerWeek",
    ]
    assert cleaned[numeric_cols].ge(0).all().all(), \
        "Negative numeric values detected"

    # Each product should have one target label per catalogue edition.
    inconsistent = (
        cleaned.groupby(["CatEdition", "ProductKey"])
        ["DiscontinuedTF"].nunique().gt(1).sum()
    )

    print("Inconsistent product-edition target labels:", inconsistent)

    if inconsistent:
        raise ValueError("Conflicting target labels require investigation")

    print("Original rows:", len(df))
    print("Cleaned rows:", len(cleaned))
    print("Status values standardised successfully")

    return cleaned


if __name__ == "__main__":
    _, catalogue, _ = load_datasets()
    clean_catalogue(catalogue)
