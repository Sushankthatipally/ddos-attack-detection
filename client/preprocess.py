"""
client/preprocess.py

Preprocessing for CLAPP anomaly detection.
Returns the raw Process–System Call matrix (PS) + labels separately,
because the paper's posterior probability computation (Eq. 4) needs
the raw counts, NOT scaled values.

For non-fuzzy-gaussian metrics an optional StandardScaler path is
provided via `scale=True`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_categorical(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Label-encode categorical columns in-place and return df."""
    le = LabelEncoder()
    for col in cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main preprocessing function
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(
    df: pd.DataFrame,
    label_col: str = "label",
    scale: bool = False,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Prepare the Process–System Call (PS) matrix and class label vector.

    Parameters
    ----------
    df        : Raw DataFrame loaded from CSV.
    label_col : Name of the target/label column. Integer (0/1) is expected;
                if strings like 'Normal'/'Attack' are found they are
                auto-encoded to 0/1.
    scale     : If True, apply StandardScaler to the feature matrix.
                Use for distance-based metrics. For 'fuzzy_gaussian' leave False.
    drop_cols : Extra columns to drop (besides the label column).

    Returns
    -------
    PS            : np.ndarray  shape (n_samples, n_features)
    labels        : np.ndarray  shape (n_samples,)  — integer class labels, or None
    feature_names : list of feature column names
    """
    df = df.copy()

    # ── 1. Resolve extra drop columns ────────────────────────────────
    if drop_cols is None:
        drop_cols = []
    always_drop = {
        "Label",
        "label",
        "Attack",
        "attack",
        "label_name",
        "attack_type",
        "difficulty",
        label_col,
    }
    remove_cols = always_drop.union(set(drop_cols))

    # ── 2. Extract labels ─────────────────────────────────────────────
    labels: Optional[np.ndarray] = None
    if label_col in df.columns:
        raw_labels = df[label_col]
        if raw_labels.dtype == object or raw_labels.dtype.name == "category":
            # String labels → encode
            le = LabelEncoder()
            labels = le.fit_transform(raw_labels.astype(str))
        else:
            labels = raw_labels.astype(int).values
    elif "Label" in df.columns:
        # Fallback — try the capitalised column
        le = LabelEncoder()
        labels = le.fit_transform(df["Label"].astype(str))

    # ── 3. Drop all non-feature columns ──────────────────────────────
    feature_cols = [c for c in df.columns if c not in remove_cols]

    # ── 4. Encode any remaining object columns (e.g. protocol_type) ──
    obj_cols = df[feature_cols].select_dtypes(include="object").columns.tolist()
    df = _encode_categorical(df, obj_cols)

    # ── 5. Build numeric PS matrix ───────────────────────────────────
    PS = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .values
        .astype(float)
    )

    # ── 6. Optional scaling ───────────────────────────────────────────
    if scale:
        scaler = StandardScaler()
        PS = scaler.fit_transform(PS)

    return PS, labels, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Train/test split that preserves class balance (stratified)
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split_stratified(
    PS: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split — keeps class proportions in train and test.
    Returns (PS_train, PS_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(
        PS, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
