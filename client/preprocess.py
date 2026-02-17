import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Drop non-numeric / identifier columns
    drop_cols = [col for col in df.columns if 'ip' in col.lower() or 'id' in col.lower()]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Keep numeric signal only; non-numeric values become NaN and are handled below.
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')

    # Replace inf and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean())

    # Remove constant columns
    df = df.loc[:, df.var() != 0]

    if df.shape[1] == 0:
        raise ValueError("No numeric feature columns remain after preprocessing.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    return X_scaled
