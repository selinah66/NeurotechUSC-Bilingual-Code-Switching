import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from scipy.stats import iqr
from config_lang import removed_cols, PROTECTED_COLUMNS


# New: Add categorical encoding
def encode_categorical(df, cat_cols=['L2 PROFICIENCY']):
    """Convert categorical columns to numeric"""
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    return df


def filter_rows(df, critical_cols=None, regex_pattern=r'^[a-zA-Z]', verbose=True):
    """Simplified row filtering without group dependencies"""
    # Language-based filtering
    df = df[df['IA_LABEL'].str.match(regex_pattern, na=False)]
    if verbose:
        print(f"Filtered English rows: {df.shape[0]}")

    # Null value filtering (skip protected columns)
    for cols in removed_cols:
        filtered_cols = [c for c in cols if c not in PROTECTED_COLUMNS]
        if not filtered_cols:
            continue

        before = len(df)
        df = df[~df[filtered_cols].isnull().all(axis=1)]
        if verbose:
            print(f"Removed {before - len(df)} rows with nulls in: {filtered_cols}")

    # Critical columns handling
    if critical_cols:
        initial = len(df)
        df = df.dropna(subset=critical_cols)
        if verbose:
            print(f"Removed {initial - len(df)} rows with missing critical cols")

    return df


def modified_scaling(df, numeric_cols, outlier_threshold=3):
    """Robust scaling with outlier clipping"""
    # Ensure numeric_cols only contains numerical columns
    numeric_cols = [col for col in numeric_cols if col in df.select_dtypes(include=np.number).columns]

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr_val = iqr(df[col])
        lower = q1 - (outlier_threshold * iqr_val)
        upper = q3 + (outlier_threshold * iqr_val)
        df[col] = df[col].clip(lower, upper)

    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def preprocess_pipeline(df, numeric_cols, critical_cols, return_steps=False):
    """Streamlined preprocessing pipeline"""
    # Stage 0: Encode categorical variables
    df = encode_categorical(df.copy())

    # Stage 1: Initial cleaning
    df_filtered = filter_rows(df, critical_cols=critical_cols)

    # Stage 2: Imputation
    df_imputed = df_filtered.copy()

    # Protected columns (essential features)
    protected = [c for c in numeric_cols if c in PROTECTED_COLUMNS]
    regular = [c for c in numeric_cols if c not in PROTECTED_COLUMNS]

    if protected:
        df_imputed[protected] = SimpleImputer(strategy='median').fit_transform(df_imputed[protected])
    if regular:
        df_imputed[regular] = SimpleImputer(strategy='mean').fit_transform(df_imputed[regular])

    # Stage 3: Scaling
    df_processed = modified_scaling(df_imputed, numeric_cols)

    if return_steps:
        return {
            'filtered': df_filtered,
            'imputed': df_imputed,
            'processed': df_processed  # Make sure this key exists
        }
    else:
        return df_processed
