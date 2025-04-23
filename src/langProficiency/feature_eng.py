import pandas as pd
import numpy as np
from config_lang import target_column
from sklearn.ensemble import RandomForestClassifier
from config_lang import RANDOM_STATE, MAX_DEPTH, N_ESTIMATORS
import matplotlib.pyplot as plt


def create_temporal_features(df):
    """Create temporal features using aggregated columns"""
    df = df.copy()

    # Create derived features
    df['regression_dwell_ratio'] = \
        df['IA_REGRESSION_PATH_DURATION'] / (df['IA_DWELL_TIME'] + 1e-6)

    df['fixation_density'] = \
        df['IA_FIXATION_COUNT'] / (df['IA_DWELL_TIME'] + 1e-6)

    df['log_first_fixation'] = np.log1p(df['IA_FIRST_FIXATION_DURATION'])

    df['saccade_speed'] = \
        df['IA_FIRST_SACCADE_AMPLITUDE'] / (df['IA_FIRST_FIXATION_DURATION'] + 1e-6)

    # Keep both original and engineered features
    return df

def aggregate_features(df, features, group_col="RECORDING_SESSION_LABEL"):
    """Updated aggregation with validated features"""
    # First validate all requested features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing features for aggregation: {missing}")

    # Define aggregations for different feature types
    agg_dict = {
        'IA_FIRST_FIXATION_DURATION': ['mean'],
        'IA_REGRESSION_PATH_DURATION': ['mean'],
        'IA_DWELL_TIME': ['mean'],
        'IA_FIXATION_COUNT': ['mean'],
        'IA_FIRST_SACCADE_AMPLITUDE': ['mean']
    }
    # Group and aggregate
    grouped = df.groupby(group_col).agg(agg_dict)
    
    # Clean up column names by removing the _mean suffix
    grouped.columns = [col for col, _ in grouped.columns]

    return grouped.reset_index()


def select_top_features(X, y, n_features=5):
    """Return selected features and their names"""
    rf = RandomForestClassifier()
    rf.fit(X, y)

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    selected_features = importance.head(n_features)['Feature'].tolist()
    return X[selected_features], selected_features  # Return names list

def prepare_train_test_data(grouped_df, cleaned_df, group_col="RECORDING_SESSION_LABEL",
                            target_col=target_column):
    """Simplified data preparation without subject IDs"""
    # Merge with target
    target = cleaned_df.drop_duplicates(group_col).set_index(group_col)[target_col]
    data = grouped_df.merge(target, left_on=group_col, right_index=True)

    # Handle class imbalance
    class_counts = data[target_col].value_counts()
    if class_counts.max() / class_counts.min() > 2:
        print(f"Applying class balancing (ratio: {class_counts.max() / class_counts.min():.1f}:1)")
        data = data.sample(frac=1, weights=data[target_col].map({k: 1 / v for k, v in class_counts.items()}))

    X = data.drop([group_col, target_col], axis=1)
    y = data[target_col]

    return X, y

