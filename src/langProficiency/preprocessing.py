import numpy as np
from scipy.stats import iqr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from config_lang import PROTECTED_COLUMNS, removed_cols

# Add categorical encoding by converting to numeric
def encode_categorical(df, cat_cols=['L2 PROFICIENCY', 'CONDITION']):
    """Convert categorical columns to numeric"""
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    return df

# Filter rows based on regex pattern and remove null values
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

# Robust scaling with outlier clipping
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

# Preprocessing pipeline
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
            'processed': df_processed
        }
    else:
        return df_processed

# Print comprehensive preprocessing metrics
def print_preprocessing_metrics(df_before, df_after, numeric_cols, steps):
    print("\nPreprocessing Metrics")
    
    # Null values
    nulls_before = df_before.isnull().sum().sum()
    nulls_after = df_after.isnull().sum().sum()
    print(f"\nNull Values:")
    print(f"Initial: {nulls_before} ({nulls_before/df_before.size:.1%})")
    print(f"Final: {nulls_after} ({nulls_after/df_after.size:.1%})")
    print(f"Reduction: {(nulls_before-nulls_after)/nulls_before:.1%}")
    
    # Class distribution
    if 'L2 PROFICIENCY' in df_after.columns:
        print("\nClass Distribution:")
        print("Before balancing:", df_before['L2 PROFICIENCY'].value_counts(normalize=True))
        print("After balancing:", df_after['L2 PROFICIENCY'].value_counts(normalize=True))
    
    # Outliers (if scaling was applied)
    if 'scaling' in steps:
        print("\nOutlier Handling:")
        total_outliers = 0
        for col in numeric_cols:
            if col in df_before.columns:
                q1 = df_before[col].quantile(0.25)
                q3 = df_before[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3*iqr
                upper = q3 + 3*iqr
                outliers = ((df_before[col] < lower) | (df_before[col] > upper)).sum()
                total_outliers += outliers
        print(f"Total outliers across all categories: {total_outliers} ({total_outliers/len(df_before):.1%})")
