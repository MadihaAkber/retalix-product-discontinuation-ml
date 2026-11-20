"""Build a leakage-aware product discontinuation dataset."""

from pathlib import Path
import pandas as pd

from src.data_ingestion import load_datasets
from src.data_cleaning import clean_catalogue

ROOT = Path(__file__).resolve().parents[1]


def build_dataset():
    products, catalogue, _ = load_datasets()
    catalogue = clean_catalogue(catalogue)
    
    # Use historical editions with complete observation windows.
    snapshot = catalogue.loc[
        catalogue["CatEdition"].between(88, 93)
        & catalogue["WeeksOut"].eq(-24)
    ].copy()

    # Remove status because it may reveal future range decisions.
    snapshot = snapshot.drop(columns=["Status"])

    # Merge product attributes without multiplying observations.
    dataset = snapshot.merge(
        products,
        on="ProductKey",
        how="left",
        validate="many_to_one"
    )

    assert len(dataset) == len(snapshot)
    assert not dataset[["CatEdition", "ProductKey"]].duplicated().any()
    assert dataset["DiscontinuedTF"].notna().all()
    assert "Status" not in dataset.columns

    output = ROOT / "data" / "processed" / "modelling_dataset.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)

    print("Modelling dataset shape:", dataset.shape)
    print("Columns:", dataset.columns.tolist())
    print("Missing values:", dataset.isna().sum().sum())
    print("Training editions:", sorted(dataset["CatEdition"].unique()))
    print("Saved to:", output)


if __name__ == "__main__":
    build_dataset()