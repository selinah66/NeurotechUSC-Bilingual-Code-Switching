from data_loader import load_data
from CS_preprocessing import label_data, clean_data
from feature_pca import apply_pca
from CS_model_training import run_loso_cv
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
        'IA_FIRST_RUN_FIXATION_COUNT', 'IA_REGRESSION_IN_COUNT', 'Switch Label'
    ]
    critical = ['IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION',
                'IA_FIRST_RUN_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE']
    cleaned = clean_data(data, numeric, critical)

    # Feature Engineering
    pca_features = [f for f in numeric if f != 'Switch Label']
    components, variance, loadings = apply_pca(cleaned, pca_features)

    # Create final dataframe for training
    pca_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(components.shape[1])])
    pca_df['Switch Label'] = cleaned['Switch Label'].values
    for col in ['L2 PROFICIENCY', 'CONDITION', 'LANGUAGE', 'RECORDING_SESSION_LABEL']:
        pca_df[col] = cleaned[col].values

    # Model Training
    accuracies = run_loso_cv(pca_df, pca_df.columns.tolist(), 'Switch Label', 'RECORDING_SESSION_LABEL', [f'PC{i+1}' for i in range(components.shape[1])])
    print("LOSO-CV Accuracy per Fold:", accuracies)
    print("Mean Accuracy:", sum(accuracies)/len(accuracies))
