"""Forward-chaining cross-validation for catalogue editions."""

import pandas as pd


def get_chronological_folds():
    df = pd.read_csv("data/processed/modelling_dataset.csv")

    # Final test edition 93 must never enter cross-validation.
    train_df = df[df["CatEdition"].between(88, 92)].copy()
    train_df = train_df.reset_index(drop=True)

    folds = []

    for validation_edition in [90, 91, 92]:
        train_idx = train_df.index[
            train_df["CatEdition"] < validation_edition
        ].to_numpy()

        validation_idx = train_df.index[
            train_df["CatEdition"] == validation_edition
        ].to_numpy()

        assert len(train_idx) > 0
        assert len(validation_idx) > 0
        assert train_df.loc[train_idx, "CatEdition"].max() < validation_edition

        folds.append((train_idx, validation_idx))

        print(
            f"Train editions 88-{validation_edition - 1}: "
            f"{len(train_idx):,} rows | "
            f"Validate edition {validation_edition}: "
            f"{len(validation_idx):,} rows"
        )

    return folds


if __name__ == "__main__":
    folds = get_chronological_folds()
    print(f"\nChronological CV validation passed: {len(folds)} folds")
