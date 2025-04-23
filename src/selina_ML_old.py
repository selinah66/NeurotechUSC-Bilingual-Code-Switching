from data_loader import load_data
from src.langProficiency.preprocessing import label_lang_prof, clean_data
from src.langProficiency.model_training import grid_search_loso
import pandas as pd
import matplotlib.pyplot as plt

file_path = '/Data/IA_data.xlsx'
raw = load_data(file_path)

# Initial processing
exclude = ['TRIAL_INDEX', 'IA_ID', 'TRIAL_LABEL', 'IA_LABEL']
data = raw.drop(columns=exclude)
data = label_lang_prof(data)

numeric = [
    'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION',
    'IA_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE', 'IA_FIXATION_COUNT',
    'IA_FIRST_RUN_FIXATION_COUNT', 'IA_REGRESSION_IN_COUNT'
]
critical = ['IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION',
            'IA_FIRST_RUN_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE']
cleaned = clean_data(data, numeric, critical, log_transform_skewed=True, outlier_clip=True,
                     low_variance_thresh=1e-3, verbose=True)

# All eye tracking features
features_to_use = [f for f in numeric if f != 'Switch Label']

# Final dataframe
cleaned['Target'] = cleaned['L2 PROFICIENCY']
model = cleaned[features_to_use + ['Target', 'RECORDING_SESSION_LABEL']]

# Model Training
results = grid_search_loso(
    data=model,
    features=model.columns.tolist(),
    label='Target',
    group_col='RECORDING_SESSION_LABEL',
    pca_cols=features_to_use,
    model_type="RandomForest"
)

# Visualize Imbalance Strategies
# Extract recall into two new columns
results[['recall_class_0', 'recall_class_1']] = pd.DataFrame(results['recall'].tolist(), index=results.index)

# Average metrics across folds for each strategy + model
grouped = results.groupby(['imbalance_strategy', 'model_type']).agg({
    'balanced_accuracy': 'mean',
    'recall_class_0': 'mean',
    'recall_class_1': 'mean'
}).reset_index()

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
metrics = ['balanced_accuracy', 'recall_class_0', 'recall_class_1']
titles = ['Balanced Accuracy', 'Recall (Minority Class)', 'Recall (Majority Class)']

for ax, metric, title in zip(axes, metrics, titles):
    for model in grouped['model_type'].unique():
        subset = grouped[grouped['model_type'] == model]
        ax.plot(subset['imbalance_strategy'].astype(str),
                subset[metric], marker='o', label=model)
    ax.set_title(title)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title='Model')

plt.suptitle("Effect of Imbalance Strategies on Classification Metrics", fontsize=16)
plt.tight_layout()
plt.show()


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