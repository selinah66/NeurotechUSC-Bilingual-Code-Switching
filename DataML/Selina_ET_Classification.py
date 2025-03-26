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

# Drop columns with >30% missing values
threshold = 30
columns_to_drop = missing_percentage[missing_percentage > threshold].index
etd_cleaned = etd.drop(columns=columns_to_drop)

# Impute remaining numeric columns
numeric_cols = etd.select_dtypes(include=['int64', 'float64']).columns
numeric_imputer = SimpleImputer(strategy='mean')
etd[numeric_cols] = numeric_imputer.fit_transform(etd[numeric_cols])

# Alternatively, impute missing values
numeric_cols = etd.select_dtypes(include=['int64', 'float64']).columns
numeric_imputer = SimpleImputer(strategy='mean')
etd_numeric_imputed = pd.DataFrame(numeric_imputer.fit_transform(etd[numeric_cols]), columns=numeric_cols)

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
