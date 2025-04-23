from data_loader import load_data
from preprocessing import preprocess_code_switching
from feature_eng import prepare_train_test_data
from model_training import train_random_forest, cross_validate_model
from config_switch import FILEPATH, numeric, critical, selected_features, RANDOM_STATE, TEST_SIZE, CATEGORICAL_COLS, target_column
from label_encoding import label_data
from src.codeSwitching.config_switch import target_column
import pandas as pd

if __name__ == "__main__":
    etd = load_data(FILEPATH)

    etd[CATEGORICAL_COLS] = etd[CATEGORICAL_COLS].astype('category')
    etd = label_data(etd)

    print("Class distribution before preprocessing:")
    print(etd['Switch Label'].value_counts())

    # Preprocess
    etd_processed = preprocess_code_switching(etd, numeric_cols=numeric, critical_cols=critical)
    print("Columns in etd_clean:", etd_processed.columns.tolist())

    print("Class distribution after preprocessing:")
    print(etd_processed['Switch Label'].value_counts())

    print(f"Unique values in target column: {etd[target_column].unique()}")

    X_train_resampled, X_test, y_train_resampled, y_test = prepare_train_test_data(
        etd_processed,
        target_col=target_column,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        apply_smote=False
    )

    # Model Training
    model = train_random_forest(X_train_resampled, y_train_resampled, X_test, y_test)
    cross_validate_model(model, X_train_resampled, y_train_resampled)
