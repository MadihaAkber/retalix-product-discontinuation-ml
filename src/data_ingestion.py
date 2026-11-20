"""Load the Retalix coursework datasets reproducibly."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Product_Discontinuation_Interview_Task"


def load_datasets():
    """Load the product, catalogue and data dictionary files."""
    paths = {
        "products": DATA_DIR / "ProductDetails.csv",
        "catalogue": DATA_DIR / "CatalogueDiscontinuation.csv",
        "dictionary": DATA_DIR / "DataDictionary.xlsx",
    }

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} dataset: {path}")

    products = pd.read_csv(paths["products"])
    catalogue = pd.read_csv(paths["catalogue"])
    dictionary = pd.read_excel(paths["dictionary"])

    return products, catalogue, dictionary


if __name__ == "__main__":
    products, catalogue, dictionary = load_datasets()

    for name, df in [
        ("ProductDetails", products),
        ("CatalogueDiscontinuation", catalogue),
        ("DataDictionary", dictionary),
    ]:
        print(f"\n{name}")
        print(f"Dimensions: {df.shape}")
        print("Column names:", df.columns.tolist())
        print("First 10 rows:")
        print(df.head(10).to_string(index=False))
