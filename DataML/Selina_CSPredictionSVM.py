import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
etd = pd.read_excel('/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching/EyeMovementData/IA_CS_data.xlsx')

# Step 2: Data Cleaning & Quality Check
# Check for missing values
print(etd.isnull().sum())

# Percentage of missing values
missing_percentage = (etd.isnull().sum() / len(etd)) * 100
print(missing_percentage)

# Drop columns with >30% missing values (Cleaning)
missing_threshold = 30
columns_to_drop = missing_percentage[missing_percentage > missing_threshold].index
print("Columns dropped:", list(columns_to_drop))  # Print the columns being dropped
etd_cleaned = etd.drop(columns=columns_to_drop)

# Impute remaining numeric columns
numeric_cols = etd_cleaned.select_dtypes(include=['int64', 'float64']).columns
numeric_imputer = SimpleImputer(strategy='mean')
etd_numeric_imputed = etd_cleaned.copy()
etd_numeric_imputed[numeric_cols] = numeric_imputer.fit_transform(etd[numeric_cols])

# Density plot for comparison
plt.figure(figsize=(10, 6))
sb.kdeplot(etd['IA_DWELL_TIME'].dropna(), label='Original', color='blue')
sb.kdeplot(etd_cleaned['IA_DWELL_TIME'], label='After Cleaning', color='green')
sb.kdeplot(etd_numeric_imputed['IA_DWELL_TIME'], label='After Imputation', color='orange')
plt.title('Density Plot of IA_DWELL_TIME')
plt.xlabel('IA_DWELL_TIME')
plt.ylabel('Density')
plt.legend()
plt.show()

# Check for outliers
from scipy import stats

def z_score_outliers(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    outliers_mask = z_scores > threshold
    return outliers_mask

outliers_mask = z_score_outliers(etd_numeric_imputed.iloc[:, 10])
print(f"Found {outliers_mask.sum()} outliers in regression path duration")

# Remove outliers
def remove_outliers(df, column, method='iqr', threshold=1.5):
    mask = z_score_outliers(df[column], threshold)

    return df[~mask]

# Apply to regression path duration
etd_cleaned = remove_outliers(etd_numeric_imputed, 'IA_REGRESSION_PATH_DURATION', method='iqr')
print(f"Original data: {len(etd)}, After removing outliers: {len(etd_cleaned)}")

import matplotlib.pyplot as plt
import seaborn as sns

# Box plot to visualize outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=etd_numeric_imputed[['IA_REGRESSION_PATH_DURATION']])
plt.title('Box Plot of Eye Tracking Metrics')
plt.ylabel('Duration (ms)')
plt.show()

# Before and after cleaning comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(etd['IA_REGRESSION_PATH_DURATION'], kde=True)
plt.title('Before Cleaning')

plt.subplot(1, 2, 2)
sns.histplot(etd_cleaned['IA_REGRESSION_PATH_DURATION'], kde=True)
plt.title('After Cleaning')
plt.tight_layout()
plt.show()

# Extract Regression Path Duration
rpd = etd_cleaned['IA_REGRESSION_PATH_DURATION']

# Validate RPD for physiological plausibility
def validate_regression_path(regression_durations, min_duration=50, max_duration=5000):
    """
    Check if all regression path durations are physiologically plausible.
    Typically between 50ms and 5000ms for reading tasks.
    """
    return (regression_durations >= min_duration) & (regression_durations <= max_duration)

is_valid = validate_regression_path(rpd)
print(f"All durations are valid: {is_valid}")

# Find and remove invalid values
invalid_values = rpd[~is_valid]
rpd_cleaned = rpd[is_valid]

print("\nShape before removing invalid values:", rpd.shape)
print("Shape after removing invalid values:", rpd_cleaned.shape)

etd = etd_cleaned

# Create labels (1 for switch points, 0 for non-switch points)
# Create new column for labelling Switching, switching = 1
etd['Switch Label'] = 0

# Assuming you have a column indicating code-switching condition
etd.loc[(etd['CONDITION'] != 'condition 1') & (etd['CONDITION'] != 'condition 2'), 'Switch Label'] = 0

# Condition 3 Labels (Insert 1 English word into Chinese sentence)
etd.loc[(etd['CONDITION'] == 'condition 3') & (etd['LANGUAGE'] == 'C'), 'Switch Label'] = 1
etd.loc[(etd['CONDITION'] == 'condition 3') & (etd['LANGUAGE'] == 'E'), 'Switch Label'] = 2 #2

# Condition 4 Labels (Insert 1 Chinese character into English sentence
etd.loc[(etd['CONDITION'] == 'condition 4') & (etd['LANGUAGE'] == 'C'), 'Switch Label'] = 1 #2
etd.loc[(etd['CONDITION'] == 'condition 4') & (etd['LANGUAGE'] == 'E'), 'Switch Label'] = 2

# Condition 5 Labels (Alteration: Chinese, then English)
etd.loc[(etd['CONDITION'] == 'condition 5') & (etd['LANGUAGE'] == 'C'), 'Switch Label'] = 1
etd.loc[(etd['CONDITION'] == 'condition 5') & (etd['LANGUAGE'] == 'E'), 'Switch Label'] = 2 #2

# Condition 6 Labels (Alteration: English, then Chinese)
etd.loc[(etd['CONDITION'] == 'condition 6') & (etd['LANGUAGE'] == 'C'), 'Switch Label'] = 1 #2
etd.loc[(etd['CONDITION'] == 'condition 6') & (etd['LANGUAGE'] == 'E'), 'Switch Label'] = 2

# Condition 7 Labels (Only nouns in English)
etd.loc[(etd['CONDITION'] == 'condition 7') & (etd['LANGUAGE'] == 'C'), 'Switch Label'] = 1
etd.loc[(etd['CONDITION'] == 'condition 7') & (etd['LANGUAGE'] == 'E'), 'Switch Label'] = 2 #2

# Condition 8 Labels (Only nouns in Chinese)
etd.loc[(etd['CONDITION'] == 'condition 8') & (etd['LANGUAGE'] == 'C'), 'Switch Label'] = 2 #2
etd.loc[(etd['CONDITION'] == 'condition 8') & (etd['LANGUAGE'] == 'E'), 'Switch Label'] = 1

# Convert to numeric type and check for any remaining NaN values
etd['Switch Label'] = pd.to_numeric(etd['Switch Label'])
print(f"Number of NaN values in Switch Label: {etd['Switch Label'].isna().sum()}")

# Define important features for detecting code switching
features = [
    'IA_FIRST_FIXATION_DURATION',      # Early lexical access
    'IA_FIRST_RUN_DWELL_TIME',         # Gaze duration
    'IA_REGRESSION_PATH_DURATION',     # Later processing difficulty
    'IA_DWELL_TIME',                   # Total reading time
    'IA_FIXATION_COUNT',               # Number of fixations
    'IA_REGRESSION_IN_COUNT',          # Regressions into the word
    'IA_REGRESSION_OUT_COUNT',         # Regressions out of the word
    'IA_SKIP'                          # Whether word was skipped
]

# Create feature vectors for each word
X = etd[features].values
y = etd['Switch Label'].values

print("Unique classes:", np.unique(y))
print("Class counts:", np.bincount(y))

# ML Model Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Train SVM model
print("\n=== Support Vector Machine ===")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluate SVM
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')

print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"SVM F1 Score: {svm_f1:.4f}")
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
