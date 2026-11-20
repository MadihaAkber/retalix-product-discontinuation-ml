"""Compare tuned models using chronological validation results."""

import pandas as pd

FILES = {
    "Logistic Regression": "outputs/tables/logistic_grid_search.csv",
    "Random Forest": "outputs/tables/random_forest_search.csv",
}


def main():
    for name, path in FILES.items():
        results = pd.read_csv(path)

        best = results.loc[results["rank_test_score"].idxmin()]

        fold_scores = [
            best[f"split{i}_test_score"]
            for i in range(3)
        ]

        print(f"\n{name}")
        print("-" * 35)

        for edition, score in zip([90, 91, 92], fold_scores):
            print(f"Validation edition {edition}: {score:.4f}")

        print(f"Mean average precision: {best['mean_test_score']:.4f}")
        print(f"Standard deviation: {best['std_test_score']:.4f}")


if __name__ == "__main__":
    main()
