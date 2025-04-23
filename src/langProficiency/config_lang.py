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

# Numeric & Critical Columns
numeric = [
    'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION',
    'IA_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE', 'IA_FIXATION_COUNT',
    'IA_FIRST_RUN_FIXATION_COUNT', 'IA_REGRESSION_IN_COUNT'
]
critical = ['IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION',
            'IA_FIRST_RUN_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE']

selected_features = ['IA_FIRST_FIXATION_DURATION', 'IA_REGRESSION_PATH_DURATION']

# decreases if add: 'IA_FIRST_RUN_FIXATION_COUNT','IA_FIXATION_COUNT'

removed_cols = [
        ['IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME'],
        ['IA_FIRST_SACCADE_AMPLITUDE']
        ]

PROTECTED_COLUMNS = [
    'IA_FIRST_FIXATION_DURATION',
    'IA_REGRESSION_PATH_DURATION',
    'IA_DWELL_TIME'
]

target_column = 'L2 PROFICIENCY'

# Train-test split
TEST_SIZE = 0.4
RANDOM_STATE = 42

# Model hyperparameters
N_ESTIMATORS = 200 #100
MAX_DEPTH = 15 #200
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 10

# Cross-validation
CV_FOLDS = 5
N_SPLITS = 5
N_REPEATS = 3