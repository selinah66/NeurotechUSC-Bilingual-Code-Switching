import os
from pathlib import Path

# Root Directory & Data Paths
ROOT_DIR = Path("/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching")
DATA_DIR = ROOT_DIR / "Data"
FILEPATH = DATA_DIR / "RawEyeMovement.xlsx"
RAW_DATA_PATH = DATA_DIR / "raw" / "eye_tracking_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "features.csv"

# Output directories
MODEL_DIR = os.path.join(DATA_DIR, '..', 'results', 'models')
PLOTS_DIR = os.path.join(DATA_DIR, '..', 'results', 'figures')

# Data Processing Settings
# Target Variable for Classification
target_column = 'L2 PROFICIENCY'

# Feature Selection
numeric = [
    'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION',
    'IA_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE', 'IA_FIXATION_COUNT',
    'IA_FIRST_RUN_FIXATION_COUNT', 'IA_REGRESSION_IN_COUNT'
]

# Critical features that should not be removed during preprocessing
critical = [
    'IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION',
    'IA_FIRST_RUN_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE'
]

# Features to protect during preprocessing
PROTECTED_COLUMNS = [
    'IA_FIRST_FIXATION_DURATION',
    'IA_REGRESSION_PATH_DURATION',
    'IA_DWELL_TIME'
]

# Feature Selection Settings
# Selected features for final model
selected_features = [
    'IA_FIRST_FIXATION_DURATION',
    'IA_REGRESSION_PATH_DURATION'
]

# Features to remove during feature selection
removed_cols = [
    ['IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME'],
    ['IA_FIRST_SACCADE_AMPLITUDE']
]

# Model Settings
# Train-test split parameters
TEST_SIZE = 0.4
RANDOM_STATE = 42

# Random Forest hyperparameters
N_ESTIMATORS = 50
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 4
MAX_FEATURES = 0.3

# Cross-validation settings
CV_FOLDS = 5
N_SPLITS = 5
N_REPEATS = 3

# Visualization Settings
skipGraphs = False