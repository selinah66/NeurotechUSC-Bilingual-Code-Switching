from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
from config_switch import exclude

def prepare_train_test_data(df, target_col="Switch Label", test_size=0.2, random_state=42, apply_smote=True):
    """Split full preprocessed data into train-test sets and optionally apply SMOTE."""

    # Separate features and target
    X = df.drop(columns=exclude + [target_col], errors='ignore')
    y = df[target_col]

    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    print("Class distribution in y_train before SMOTE:")
    print(y_train.value_counts())

    print("Class distribution in y_test:")
    print(y_test.value_counts())

    X_train_numeric = X_train.select_dtypes(include=['number'])
    X_test_numeric = X_train.select_dtypes(include=['number'])
    X_train[X_train_numeric.columns] = X_train_numeric.fillna(X_train_numeric.median())
    X_test[X_test_numeric.columns] = X_test_numeric.fillna(X_test_numeric.median())

    if y_train.nunique() > 1:
        if apply_smote:
            smote = SMOTE(sampling_strategy='auto', random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            print("Class distribution in y_train after SMOTE:")
            print(pd.Series(y_train_resampled).value_counts())

            return X_train_resampled, X_test, y_train_resampled, y_test
        else:
            return X_train, X_test, y_train, y_test
    else:
        raise ValueError("y_train contains only one class. Please check your dataset.")