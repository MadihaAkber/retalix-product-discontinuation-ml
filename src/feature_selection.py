"""Document and validate the model feature selection."""

from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

RETAINED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXCLUDED_FEATURES = {
    "DiscontinuedTF": "Target variable, not a predictor",
    "ProductKey": "Identifier; may encourage memorisation",
    "Status": "Potential leakage from future range decisions",
    "CatEdition": "Reserved for chronological splitting",
    "WeeksOut": "Constant at the Week -24 prediction snapshot",
}


def validate_features():
    """Check that excluded features cannot enter the model."""
    overlap = set(RETAINED_FEATURES) & set(EXCLUDED_FEATURES)
    assert not overlap, f"Excluded features retained: {overlap}"

    assert len(RETAINED_FEATURES) == len(set(RETAINED_FEATURES))

    print("Retained features:", RETAINED_FEATURES)
    print("\nExcluded features and reasons:")

    for feature, reason in EXCLUDED_FEATURES.items():
        print(f"{feature}: {reason}")

    print("\nFeature selection validation passed!")


if __name__ == "__main__":
    validate_features()
