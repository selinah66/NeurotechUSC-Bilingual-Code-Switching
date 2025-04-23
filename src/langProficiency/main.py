from data_loader import load_data
from preprocessing import preprocess_pipeline
from feature_pca import apply_pca
from feature_eng import aggregate_features, create_temporal_features, select_top_features, prepare_train_test_data
from visualizations import doGraphs
from sklearn.model_selection import train_test_split
from model_training import train_random_forest, tune_random_forest, plot_feature_importance, evaluate_model
from config_lang import FILEPATH, numeric, critical, RANDOM_STATE, TEST_SIZE

if __name__ == "__main__":
    # 1. Load and visualize raw data
    etd = load_data(FILEPATH)
    doGraphs(etd)

    # 2. Preprocess data
    etd_clean = preprocess_pipeline(etd, numeric, critical)
    print("Cleaned data columns:", etd_clean.columns.tolist())

    # 3. Feature engineering
    features_grouped = aggregate_features(etd_clean, numeric)
    engineered_data = create_temporal_features(features_grouped)
    print("Engineered features:", engineered_data.columns.tolist())

    # 4. Feature selection
    X, y = prepare_train_test_data(features_grouped, etd_clean)
    X_selected, selected_features = select_top_features(X, y)

    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 6. Model training and evaluation
    best_model = tune_random_forest(X_selected, y)
    plot_feature_importance(best_model, selected_features)

    final_model = train_random_forest(X_train, y_train, X_test, y_test)
    evaluate_model(final_model, X_selected, y)
