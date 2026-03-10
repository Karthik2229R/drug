"""
data_loader.py — Load the DDI dataset, inspect it, encode labels, and compute
class weights for handling imbalanced interaction types.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from src.utils import CSV_PATH


def load_dataset(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load the DDI CSV dataset and perform basic inspection.
    Prints shape, dtypes, missing values, and class distribution.
    """
    df = pd.read_csv(csv_path)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    # Flag any issues
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        print(f"\n⚠ WARNING: {total_missing} missing values found!")
        df = df.dropna()
        print(f"  → Dropped rows with missing values. New shape: {df.shape}")
    else:
        print("\n✓ No missing values.")

    # Check for duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"⚠ WARNING: {n_dupes} duplicate rows found — removing.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("✓ No duplicate rows.")

    # Class distribution
    print(f"\nInteraction type distribution ({df['interaction_type'].nunique()} classes):")
    print(df["interaction_type"].value_counts().to_string())

    return df


def build_drug_index(df: pd.DataFrame) -> dict:
    """
    Build a mapping from drug ID → unique integer index.
    Covers all drugs appearing in either drug1_id or drug2_id.
    Returns:
        drug_to_idx : dict mapping drug_id string → int index
    """
    all_drugs = pd.concat([df["drug1_id"], df["drug2_id"]]).unique()
    all_drugs.sort()
    drug_to_idx = {drug: idx for idx, drug in enumerate(all_drugs)}
    print(f"\nUnique drugs: {len(drug_to_idx)}")
    return drug_to_idx


def build_drug_name_map(df: pd.DataFrame) -> dict:
    """
    Build a mapping from drug_name → drug_id for the Streamlit UI.
    If a name maps to multiple IDs, keep the first occurrence.
    """
    name_to_id = {}
    for _, row in df[["drug1_id", "drug1_name"]].drop_duplicates().iterrows():
        name_to_id.setdefault(row["drug1_name"], row["drug1_id"])
    for _, row in df[["drug2_id", "drug2_name"]].drop_duplicates().iterrows():
        name_to_id.setdefault(row["drug2_name"], row["drug2_id"])
    return name_to_id


def encode_labels(df: pd.DataFrame):
    """
    Encode interaction_type strings to integer labels.
    Returns:
        labels      : np.ndarray of integer labels
        le          : fitted LabelEncoder (for inverse_transform later)
        num_classes : int — number of unique interaction types
    """
    le = LabelEncoder()
    labels = le.fit_transform(df["interaction_type"].values)
    num_classes = len(le.classes_)
    print(f"Encoded {num_classes} interaction types → integer labels [0, {num_classes - 1}]")
    return labels, le, num_classes


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    Classes with fewer samples get higher weight to counter imbalance.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    # Inverse frequency, normalized so weights sum to num_classes
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    print(f"\nClass weights (min={weights.min():.4f}, max={weights.max():.4f}):")
    return torch.tensor(weights, dtype=torch.float32)


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_dataset()
    drug_to_idx = build_drug_index(df)
    labels, le, num_classes = encode_labels(df)
    class_weights = compute_class_weights(labels, num_classes)
    print("\nLabel classes:", list(le.classes_))
    print("Class weights:", class_weights)
