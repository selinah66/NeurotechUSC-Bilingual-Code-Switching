import numpy as np
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2OGradientBoostingEstimator
from sklearn.model_selection import GroupKFold
import pandas as pd

def run_loso_cv(data, features, label, group_col, pca_cols):
    h2o.init()
    accuracies = []
    groups = data[group_col]
    group_kfold = GroupKFold(n_splits=5)

    for train_idx, test_idx in group_kfold.split(data, data[label], groups=groups):
        train = data.iloc[train_idx].drop(columns=[group_col])
        test = data.iloc[test_idx].drop(columns=[group_col])
        h2o_train = h2o.H2OFrame(train)
        h2o_test = h2o.H2OFrame(test)

        for target_col in [label]:
            h2o_train[target_col] = h2o_train[target_col].asfactor()
            h2o_test[target_col] = h2o_test[target_col].asfactor()

        grid = H2OGridSearch(
            model = H2OGradientBoostingEstimator(balance_classes=True, seed=42, stopping_metric="AUC", stopping_rounds=3,
                                                 stopping_tolerance=0.001),
            hyper_params={
                'ntrees': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_rows': [2, 5, 10],
                'sample_rate': [0.7, 0.8, 1.0],
                'col_sample_rate_per_tree': [0.7, 0.9, 1.0],
                'class_sampling_factors': [
                    [1.0, 1.0],  # baseline (no change)
                    [1.67, 0.94],  # boost class 0 slightly
                    [1.1, 1.0],
                    [1.2, 1.0],
                    [1.3, 1.0],
                    [1.4, 1.0]],  # highest you might go before collapse
            },
            search_criteria={'strategy': "RandomDiscrete", 'max_models': 20, 'seed': 1}
        )

        print(train[label].value_counts(), "\n", test[label].value_counts())

        grid.train(x=pca_cols, y=label, training_frame=h2o_train, validation_frame=h2o_test)

        sorted_grid = grid.get_grid(sort_by='mean_per_class_error', decreasing=False)
        best_model = sorted_grid.models[0]  # Extract the actual model
        perf = best_model.model_performance(h2o_test)

        # Compute confusion matrix and accuracy
        conf_matrix_raw = perf.confusion_matrix()
        conf_matrix_df = conf_matrix_raw.table.as_data_frame()

        # Extract values safely — look for the 'Count' column
        if 'Count' in conf_matrix_df.columns:
            # Classic confusion matrix with counts
            conf_matrix_values = conf_matrix_df.pivot(index='Actual', columns='Predicted',
                                                      values='Count').values.astype(int)
        else:
            # Backup in case format is different (e.g., cell values are already in the table body)
            matrix_only = conf_matrix_df.iloc[:-1, 1:-1]  # Skip labels and totals
            conf_matrix_values = matrix_only.astype(float).values  # Use float just in case

        print(conf_matrix_df)

        # Compute accuracy
        acc = np.trace(conf_matrix_values) / np.sum(conf_matrix_values)
        accuracies.append(acc)

    h2o.cluster().shutdown()
    return accuracies