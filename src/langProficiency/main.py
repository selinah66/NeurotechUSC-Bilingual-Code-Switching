from data_loader import load_data
from preprocessing import preprocess_pipeline
from feature_eng import aggregate_features, create_temporal_features, select_top_features, prepare_train_test_data
from src.codeSwitching.config_switch import MAX_DEPTH
from src.langProficiency.visualizations import plot_preprocessing_stages
from visualizations import doGraphs, plot_feature_correlation, plot_feature_importance_analysis, visualize_decision_trees
from sklearn.model_selection import train_test_split
from model_training import train_random_forest, tune_random_forest, evaluate_model
from config_lang import FILEPATH, numeric, critical, RANDOM_STATE, TEST_SIZE

if __name__ == "__main__":
    # 1. Load and visualize raw data
    etd = load_data(FILEPATH)
    doGraphs(etd, skipGraphs=True)

    # 2. Preprocess data
    etd_clean_steps = preprocess_pipeline(etd, numeric, critical, return_steps=True)
    print("Available dictionary keys:", list(etd_clean_steps.keys()))
    if 'processed' in etd_clean_steps:
        print("Cleaned data columns:", etd_clean_steps['processed'].columns.tolist())
    else:
        # Try to find the final processed data
        keys = list(etd_clean_steps.keys())
        if keys:
            print("Using alternative key:", keys[-1])
            print("Cleaned data columns:", etd_clean_steps[keys[-1]].columns.tolist())
        else:
            print("No data returned from preprocessing pipeline")

    plot_preprocessing_stages(etd_clean_steps, 'IA_FIXATION_COUNT')

    # 3. Feature engineering
    features_grouped = aggregate_features(etd_clean_steps['processed'], numeric)
    print("\nFeatures after aggregation:")
    print(features_grouped.columns.tolist())
    
    engineered_data = create_temporal_features(features_grouped)
    print("\nFeatures after engineering:")
    print(engineered_data.columns.tolist())

    # 4. Feature selection
    X, y = prepare_train_test_data(engineered_data, etd_clean_steps['processed'])
    print("\nFeatures in X:")
    print(X.columns.tolist())
    
    X_selected, selected_features = select_top_features(X, y)
    print("\nSelected Features (Top 5):")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    # Visualize feature correlations and importance
    plot_feature_correlation(X, X.columns)  # Show correlation for all features
    plot_feature_importance_analysis(X, y, selected_features)

    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 6. Model training and evaluation
    best_model = tune_random_forest(X_selected, y)
    test_score = best_model.score(X_test, y_test)
    print(f"Final test score: {test_score:.2f}")

    final_model = train_random_forest(X_train, y_train, X_test, y_test)
    evaluate_model(final_model, X_selected, y)
    print(f"Train Accuracy: {final_model.score(X_train, y_train):.2f}")
    print(f"Test Accuracy: {final_model.score(X_test, y_test):.2f}")

    # Visualize decision trees
    visualize_decision_trees(final_model, selected_features, max_depth=MAX_DEPTH, n_trees=3)
