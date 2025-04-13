import numpy as np
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2ORandomForestEstimator
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

        for col in [label, 'L2 PROFICIENCY', 'CONDITION', 'LANGUAGE']:
            h2o_train[col] = h2o_train[col].asfactor()
            h2o_test[col] = h2o_test[col].asfactor()

        grid = H2OGridSearch(
            model = H2ORandomForestEstimator(balance_classes=True, seed=42, stopping_rounds=3,
                                           stopping_metric="misclassification", stopping_tolerance=0.001),
            hyper_params={
                'ntrees': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_rows': [2, 5, 10],
                'sample_rate': [0.7, 0.8, 0.9],
                'col_sample_rate_per_tree': [0.6, 0.8, 1.0]
            },
            search_criteria={'strategy': "RandomDiscrete", 'max_models': 20, 'seed': 42}
        )

        grid.train(x=pca_cols, y=label, training_frame=h2o_train, validation_frame=h2o_test)

        best_model = grid.get_grid(sort_by='accuracy', decreasing=True).models[0]
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