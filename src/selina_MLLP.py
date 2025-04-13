from data_loader import load_data
from LP_preprocessing import label_data, clean_data
from LP_model_training import run_loso_cv
import pandas as pd

if __name__ == "__main__":
    file_path = '/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching/Data/IA_data.xlsx'
    raw = load_data(file_path)

    # Initial processing
    exclude = ['TRIAL_INDEX', 'IA_ID', 'TRIAL_LABEL', 'IA_LABEL']
    data = raw.drop(columns=exclude)
    data = label_data(data)

    numeric = [
        'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION',
        'IA_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE', 'IA_FIXATION_COUNT', 'IA_SKIP',
        'IA_FIRST_RUN_FIXATION_COUNT', 'IA_REGRESSION_IN_COUNT'
    ]
    critical = ['IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION',
                'IA_FIRST_RUN_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE']
    cleaned = clean_data(data, numeric, critical)

    # Skip PCA
    features_to_use = [f for f in numeric if f != 'Switch Label']  # Keep all original eye-tracking features

    # Final dataframe
    cleaned['Target'] = cleaned['L2 PROFICIENCY']
    model_df = cleaned[features_to_use + ['Target', 'RECORDING_SESSION_LABEL']]

    # Model Training
    accuracies = run_loso_cv(
        model_df,
        model_df.columns.tolist(),
        'Target',
        'RECORDING_SESSION_LABEL',
        numeric
    )
