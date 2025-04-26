import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config_lang import (
    FILEPATH, MAX_DEPTH, numeric, critical, 
    RANDOM_STATE, TEST_SIZE
)
from data_loader import load_data
from feature_eng import (
    aggregate_features, create_temporal_features, 
    prepare_train_test_data, select_top_features
)
from model_training import (
    evaluate_model, train_random_forest, 
    tune_random_forest
)
from preprocessing import (
    preprocess_pipeline, print_preprocessing_metrics
)
from visualizations import (
    doGraphs, plot_descriptive_stats_comparison,
    plot_feature_correlation, plot_feature_importance_analysis,
    plot_preprocessing_stages, visualize_decision_trees
)

if __name__ == "__main__":
    # Load and visualize raw data
    etd = load_data(FILEPATH)
    doGraphs(etd, skipGraphs=False)

    # Preprocess data
    etd_clean_steps = preprocess_pipeline(etd, numeric, critical, return_steps=True)
    
    # Print preprocessing metrics
    print_preprocessing_metrics(
        etd, 
        etd_clean_steps['processed'], 
        numeric,
        steps=['encoding', 'filtering', 'imputation', 'scaling']
    )
    
    # Plot how features (Fixation Count) changes after each preprocessing stage
    plot_preprocessing_stages(etd_clean_steps, 'IA_FIXATION_COUNT')
    
    # Plot comparison of descriptive statistics before and after preprocessing
    plot_descriptive_stats_comparison(etd_clean_steps, numeric)

    # Feature engineering by aggregating features and creating temporal features
    features_grouped = aggregate_features(etd_clean_steps['processed'], numeric)
    print("\nFeatures after aggregation:", features_grouped.columns.tolist())
    
    engineered_data = create_temporal_features(features_grouped)
    print("\nFeatures after engineering:", engineered_data.columns.tolist())

    # Feature selection
    X, y = prepare_train_test_data(engineered_data, etd_clean_steps['processed'])
    print("\nFeatures in X:", X.columns.tolist())
    
    X_selected, selected_features = select_top_features(X, y)
    print("\nSelected Features (Top 5):")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    # Visualize feature correlations and importance
    plot_feature_correlation(X, X.columns)
    plot_feature_importance_analysis(X, y, selected_features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Model training and evaluation
    # Tune Random Forest classifier for best performance
    best_model = tune_random_forest(X_selected, y)
    test_score = best_model.score(X_test, y_test)
    print(f"Final test score: {test_score:.2f}")

    # Train final RF model
    final_model = train_random_forest(X_train, y_train, X_test, y_test)

    # Evaluate final RF model
    evaluate_model(final_model, X_selected, y)
    print(f"Train Accuracy: {final_model.score(X_train, y_train):.2f}")
    print(f"Test Accuracy: {final_model.score(X_test, y_test):.2f}")

    # Visualize decision trees
    visualize_decision_trees(final_model, selected_features, max_depth=MAX_DEPTH, n_trees=3)
