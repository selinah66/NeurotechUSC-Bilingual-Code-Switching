import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
etd = pd.read_excel('/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching/EyeMovementData/IA_data.xlsx')

# Step 2: Initial Data Visualization
# Plot 1. Scatter Plot: Fixation Count vs. First Saccade Amplitude
plt.figure(figsize=(8, 6))
plt.scatter(etd.iloc[:, 13], etd.iloc[:, 12])  # Assuming Column 14 is Fixation Count and Column 13 is First Saccade Amplitude
plt.xlabel('Fixation Count')
plt.ylabel('First Saccade Amplitude')
plt.title('Fixation Count vs. Saccade Amplitude')
plt.show()

# Plot 2. Bar Chart: Average fixation durations
avg_first_fixation = etd.iloc[:, 7].mean()  # Assuming Column 8 is the first fixation duration
avg_second_fixation = etd.iloc[:, 8].mean()

# Plot bar chart
plt.bar(['First Fixation', 'Second Fixation'], [avg_first_fixation, avg_second_fixation])
plt.xlabel('Fixation Type')
plt.ylabel('Average Duration (s)')
plt.title('Average Fixation Durations')
plt.show()

# Plot 3. Stacked Bar Chart: Regression Paths vs. First Run Dwell Time
# Calculate sums for each type of gaze behavior
regression_path_time = etd.iloc[:, 10].sum()  # Assuming Column 11 is Regression Path Duration
first_run_dwell_time = etd.iloc[:, 9].sum()  # Assuming Column 10 is First Run Dwell Time

# Plot stacked bar chart
plt.bar(['Regression Paths', 'First Run Dwell Time'], [regression_path_time, first_run_dwell_time])
plt.xlabel('Gaze Behavior')
plt.ylabel('Total Time (s)')
plt.title('Gaze Behavior Comparison')
plt.show()

# Step 3: Data Cleaning & Quality Check
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

# Extract Fixation Count (column 14)
fix = etd_cleaned['IA_FIXATION_COUNT']

# Check for Short Fixation Events (<50 ms)
def short_fixations(fixation_count, threshold=50):
    """
    Check fixation counts are shorter than 50 ms, and remove them.
    """
    return (fixation_count < threshold)

valid_fixation = short_fixations(fix)
print(f"All fixations are valid: {valid_fixation}")