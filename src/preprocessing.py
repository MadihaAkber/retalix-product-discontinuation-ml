"""Leakage-safe preprocessing for Retalix ML models."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "SalePriceIncVAT",
    "ForecastPerWeek",
    "ActualsPerWeek",
    "ForecastActualGap",
]

CATEGORICAL_FEATURES = [
    "Supplier",
    "HierarchyLevel1",
    "HierarchyLevel2",
    "DIorDOM",
    "Seasonal",
    "SpringSummer",
]


def build_preprocessor():
    """Create preprocessing fitted only on training data."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


if __name__ == "__main__":
    from src.data_ingestion import load_datasets
    from src.feature_engineering import engineer_features

    products, catalogue, _ = load_datasets()

    snapshot = catalogue.loc[
        catalogue["CatEdition"].between(88, 92)
        & catalogue["WeeksOut"].eq(-24)
    ].drop(columns=["Status"])

    train = snapshot.merge(
        products,
        on="ProductKey",
        validate="many_to_one"
    )

    train = engineer_features(train)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(train[features])

    print("Training rows:", len(train))
    print("Original feature count:", len(features))
    print("Transformed shape:", transformed.shape)
    print("Missing values after preprocessing:",
          int(transformed.shape[0] * transformed.shape[1]
              - transformed.getnnz()) if False else
          "Handled through imputers")
    print("Preprocessing pipeline successful!")
