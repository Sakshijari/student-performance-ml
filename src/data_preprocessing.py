"""
Data preprocessing pipeline for Student Performance prediction.
Handles loading, cleaning, target derivation, encoding, scaling, and stratified split.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Reproducibility
RANDOM_STATE = 42


def load_data(data_path: str) -> pd.DataFrame:
    """Load student performance CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    return df


def derive_target(df: pd.DataFrame, score_cols=None, threshold: float = 60.0) -> pd.DataFrame:
    """
    Derive binary target: pass (1) if average of score_cols >= threshold, else fail (0).
    """
    if score_cols is None:
        score_cols = ["math score", "reading score", "writing score"]
    df = df.copy()
    df["average_score"] = df[score_cols].mean(axis=1)
    df["pass"] = (df["average_score"] >= threshold).astype(int)
    return df


def get_feature_columns():
    """Categorical and numerical feature names for the dataset."""
    categorical = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
    ]
    numerical = ["math score", "reading score", "writing score"]
    return categorical, numerical


def build_preprocessor(categorical_cols, numerical_cols):
    """Build ColumnTransformer for encoding and scaling."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
):
    """
    Full preprocessing: derive target, encode, scale, stratified split.
    Returns X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names.
    """
    df = derive_target(df)
    categorical_cols, numerical_cols = get_feature_columns()
    X = df[categorical_cols + numerical_cols]
    y = df["pass"]

    # First split: train+val vs test (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out(categorical_cols)
    feature_names = list(numerical_cols) + list(cat_features)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train.values,
        "y_val": y_val.values,
        "y_test": y_test.values,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    # Quick test
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "StudentsPerformance.csv")
    df = load_data(data_path)
    df = derive_target(df)
    print("Target distribution (pass):", df["pass"].value_counts())
    result = preprocess_and_split(df)
    print("Shapes:", result["X_train"].shape, result["X_val"].shape, result["X_test"].shape)
    print("Feature names count:", len(result["feature_names"]))
