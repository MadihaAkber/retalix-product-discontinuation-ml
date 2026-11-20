"""Create a chronological train/test split."""

import pandas as pd

from src.feature_engineering import engineer_features
from src.feature_selection import RETAINED_FEATURES


def get_train_test_data():
    df = pd.read_csv("data/processed/modelling_dataset.csv")
    df = engineer_features(df)

    train = df[df["CatEdition"].between(88, 92)].copy()
    test = df[df["CatEdition"].eq(93)].copy()

    X_train = train[RETAINED_FEATURES]
    y_train = train["DiscontinuedTF"].astype(int)

    X_test = test[RETAINED_FEATURES]
    y_test = test["DiscontinuedTF"].astype(int)

    assert len(X_train) > 0 and len(X_test) > 0
    assert set(train["CatEdition"]).isdisjoint(set(test["CatEdition"]))
    assert list(X_train.columns) == list(X_test.columns)
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data()

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)
    print("Training discontinuation rate:", round(y_train.mean(), 4))
    print("Testing discontinuation rate:", round(y_test.mean(), 4))
    print("Chronological split validation passed!")
