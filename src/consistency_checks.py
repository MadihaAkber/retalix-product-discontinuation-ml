"""Identify potential inconsistencies in Retalix data."""

from src.data_ingestion import load_datasets


def main():
    products, catalogue, _ = load_datasets()

    print("=== PRODUCT CHECKS ===")
    print("Duplicate ProductKeys:",
          products["ProductKey"].duplicated().sum())

    print("\nDIorDOM categories:")
    print(products["DIorDOM"].value_counts(dropna=False))

    print("\n=== CATALOGUE CHECKS ===")
    print("Negative prices:",
          (catalogue["SalePriceIncVAT"] < 0).sum())

    print("Negative forecasts:",
          (catalogue["ForecastPerWeek"] < 0).sum())

    print("Negative actual sales:",
          (catalogue["ActualsPerWeek"] < 0).sum())

    print("\nStatus categories:")
    print(catalogue["Status"].value_counts(dropna=False))

    print("\nCatalogue editions:")
    print(sorted(catalogue["CatEdition"].unique()))

    print("\nWeeksOut range:",
          catalogue["WeeksOut"].min(),
          catalogue["WeeksOut"].max())

    print("\nUnmatched product records:",
          (~catalogue["ProductKey"].isin(
              products["ProductKey"]
          )).sum())


if __name__ == "__main__":
    main()
