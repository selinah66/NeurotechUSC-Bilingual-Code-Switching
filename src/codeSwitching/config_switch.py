import os
from pathlib import Path

# Root directory
ROOT_DIR = Path("/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching")

# Data paths
DATA_DIR = ROOT_DIR / "Data"
FILEPATH = DATA_DIR / "RawEyeMovement.xlsx"
RAW_DATA_PATH = DATA_DIR / "raw" / "eye_tracking_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "features.csv"

skipGraphs = False

# Model output
MODEL_DIR = os.path.join(DATA_DIR, '..', 'results', 'models')
PLOTS_DIR = os.path.join(DATA_DIR, '..', 'results', 'figures')

exclude = ['TRIAL_INDEX', 'IA_ID', 'TRIAL_LABEL', 'IA_LABEL', 'RECORDING_SESSION_LABEL', 'L2 PROFICIENCY', 'CONDITION', 'LANGUAGE']

# Numeric & Critical Columns
numeric = [
    'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION',
    'IA_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE', 'IA_FIXATION_COUNT',
    'IA_FIRST_RUN_FIXATION_COUNT', 'IA_REGRESSION_IN_COUNT'
]
critical = ['IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION',
            'IA_FIRST_RUN_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE']

selected_features = [
    'IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION',
    'IA_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE']

CATEGORICAL_COLS = ['L2 PROFICIENCY', 'CONDITION', 'LANGUAGE']

removed_cols = [
        ['IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME'],
        ['IA_FIRST_SACCADE_AMPLITUDE']
    ]

# decreases if add: 'IA_FIRST_RUN_FIXATION_COUNT','IA_FIXATION_COUNT'

target_column = 'Switch Label'

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model hyperparameters
N_ESTIMATORS = 50
MAX_DEPTH = 10
min_samples_split = 10
min_samples_leaf = 5

# Cross-validation
CV_FOLDS = 5
