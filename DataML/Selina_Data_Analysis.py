import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
etd = pd.read_csv('/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching/EyeMovementData/IA_data.csv')

# Step 2: Initial Data Visualization
# Plot scatter plot
plt.scatter(etd.iloc[:, 13], etd.iloc[:, 12])  # Assuming Column 14 is Fixation Count and Column 13 is First Saccade Amplitude
plt.xlabel('Fixation Count')
plt.ylabel('First Saccade Amplitude')
plt.title('Fixation Count vs. Saccade Amplitude')
# plt.show()

# Calculate average fixation durations
avg_first_fixation = etd.iloc[:, 7].mean()  # Assuming Column 8 is the first fixation duration
avg_second_fixation = etd.iloc[:, 8].mean()

# Plot bar chart
plt.bar(['First Fixation', 'Second Fixation'], [avg_first_fixation, avg_second_fixation])
plt.xlabel('Fixation Type')
plt.ylabel('Average Duration (s)')
plt.title('Average Fixation Durations')
# plt.show()

# Stacked Bar Chart: Regression Paths
# Calculate sums for each type of gaze behavior
regression_path_time = etd.iloc[:, 10].sum()  # Assuming Column 11 is Regression Path Duration
first_run_dwell_time = etd.iloc[:, 9].sum()  # Assuming Column 10 is First Run Dwell Time

# Plot stacked bar chart
plt.bar(['Regression Paths', 'First Run Dwell Time'], [regression_path_time, first_run_dwell_time])
plt.xlabel('Gaze Behavior')
plt.ylabel('Total Time (s)')
plt.title('Gaze Behavior Comparison')
# plt.show()

# Step 3: Data Cleaning & Quality Check
# Check for missing values
print(etd.isnull().sum())

# Remove missing values (if appropriate)
etd_cleaned = etd.dropna()

print(etd_cleaned.isnull().sum())

# Alternatively, impute missing values
numeric_cols = etd.select_dtypes(include=['int64', 'float64']).columns
numeric_imputer = SimpleImputer(strategy='mean')
etd_numeric_imputed = pd.DataFrame(numeric_imputer.fit_transform(etd[numeric_cols]), columns=numeric_cols)

# Visualize the distribution before and after imputation to ensure it makes sense
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(etd['IA_DWELL_TIME'].dropna(), bins=20)
plt.title('Original')

plt.subplot(1, 2, 1)
plt.hist(etd_cleaned['IA_DWELL_TIME'], bins=20)
plt.title('After Cleaning')
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(etd_numeric_imputed['IA_DWELL_TIME'], bins=20)
plt.title('After Imputation')
plt.show()

# Check for outliers (e.g., using z-score method)
from scipy import stats
z_scores = stats.zscore(etd['IA_DWELL_TIME'])  # Replace 'DwellTime' with your column of interest
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=0)  # Common threshold for outliers is 3 standard deviations
etd_filtered = etd[filtered_entries]