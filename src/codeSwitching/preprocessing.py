import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from config_switch import removed_cols
import pandas as pd

def get_removed_rows(df, etd_cols):
    removed_indices = set()
    for cols in etd_cols:
        all_null_mask = df[cols].isnull().all(axis=1)
        removed_indices.update(df[all_null_mask].index.tolist())
    return df.loc[sorted(removed_indices)]

def filter_rows(df, critical_cols=None, verbose=True):

    if removed_cols:
        for cols in removed_cols:
            before = len(df)
            df = df[~df[cols].isnull().all(axis=1)]
            after = len(df)
            if verbose and before != after:
                print(f"Filtered {before - after} rows with all-null values in columns: {cols}")

    if critical_cols:
        initial_count = len(df)
        df = df.dropna(subset=critical_cols)
        if verbose:
            print(f"Removed {initial_count - len(df)} rows with missing critical cols")

    return df

def drop_missing_columns(df, numeric_cols, threshold=30, verbose=True):
    missing_percent = df[numeric_cols].isna().mean() * 100
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    if verbose and cols_to_drop:
        print(f"Dropped columns with >{threshold}% missing: {cols_to_drop}")
    return df

def impute_missing_numeric(df, numeric_cols, verbose=True):
    imputed_cols = []
    for col in numeric_cols:
        if df[col].isna().any():
            imputer = SimpleImputer(strategy='mean')
            df[[col]] = imputer.fit_transform(df[[col]])
            df[f"{col}_was_missing"] = 0
            imputed_cols.append(col)
    if verbose and imputed_cols:
        print("Imputed missing values using mean for:")
        for col in imputed_cols:
            print(f" - {col}")
    return df

def handle_data_distribution(df, numeric_cols,
                             strategy='transform',
                             skew_thresh=1,
                             clip_percentiles=(1, 99),
                             verbose=True):
    if strategy in ('transform', 'both'):
        skewed_cols = [col for col in numeric_cols if df[col].skew() > skew_thresh]
        for col in skewed_cols:
            df[col] = np.log1p(df[col])
        if verbose and skewed_cols:
            print(f"Log-transformed skewed cols: {skewed_cols}")

    if strategy in ('clip', 'both'):
        for col in numeric_cols:
            low, high = np.percentile(df[col], clip_percentiles)
            df[col] = df[col].clip(lower=low, upper=high)
        if verbose:
            print(f"Clipped to {clip_percentiles[0]}%-{clip_percentiles[1]}% range")
    return df

def remove_outliers_zscore(df, numeric_cols, threshold=3, verbose=True):
    original_rows = len(df)
    outlier_mask = np.zeros(len(df), dtype=bool)

    for col in numeric_cols:
        if df[col].notna().all():
            z_scores = np.abs(stats.zscore(df[col]))
            col_mask = z_scores > threshold
            outlier_mask |= col_mask
            if verbose:
                print(f"Found {col_mask.sum()} outliers in {col}")

    df_clean = df[~outlier_mask]
    if verbose:
        print(f"\nRemoved {original_rows - len(df_clean)} rows containing outliers")
        print(f"New dataset shape: {df_clean.shape}")

    return df_clean

def standardize_numeric(df, numeric_cols, verbose=True):
    df_std = df.copy()
    scaler = StandardScaler()
    df_std[numeric_cols] = scaler.fit_transform(df_std[numeric_cols])
    if verbose:
        print("Standardized numeric columns.")
    return df_std

def preprocess_code_switching(df, numeric_cols, critical_cols, verbose=True):

    # Step 1: Filtering
    df = filter_rows(df, critical_cols=critical_cols, verbose=verbose)

    print("Class distribution after filtering:")
    print(pd.Series(df['Switch Label']).value_counts())

    # Step 2: Drop high-missing columns
    df = drop_missing_columns(df, numeric_cols=numeric_cols, verbose=verbose)
    print("Class distribution after dropping:")
    print(pd.Series(df['Switch Label']).value_counts())

    # Step 3: Impute missing
    df = impute_missing_numeric(df, numeric_cols=numeric_cols, verbose=verbose)
    print("Class distribution after imputing:")
    print(pd.Series(df['Switch Label']).value_counts())

    # Step 4: Outlier removal
    df = remove_outliers_zscore(df, numeric_cols=numeric_cols, verbose=verbose)
    print("Class distribution after removing outliers:")
    print(pd.Series(df['Switch Label']).value_counts())

    # Step 5: Standardization
    df = standardize_numeric(df, numeric_cols=numeric_cols, verbose=verbose)

    return df