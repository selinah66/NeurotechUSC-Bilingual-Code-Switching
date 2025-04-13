import numpy as np
from scipy import stats

def label_data(features):
    features['L2 PROFICIENCY'] = features['L2 PROFICIENCY'].map({'L': 0, 'H': 1}).astype(int)

    return features

def clean_data(features, numeric_cols, critical_cols):
    # Drop numeric cols with >30% missing
    missing_pct = features[numeric_cols].isna().mean() * 100
    features = features.drop(columns=missing_pct[missing_pct > 30].index)
    features = features.dropna(subset=critical_cols)

    # Outlier removal
    numeric_cols = [col for col in numeric_cols if col in features.columns]
    z_scores = np.abs(stats.zscore(features[numeric_cols]))
    mask = (z_scores > 3).any(axis=1)
    clean_idx = features[numeric_cols][~mask].index
    return features.loc[clean_idx]