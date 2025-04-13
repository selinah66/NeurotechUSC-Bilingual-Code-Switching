import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
import h2o
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from h2o.estimators import H2ORandomForestEstimator

# Step 1. Load the dataset
file_path = '/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching/EyeMovementData/IA_data.xlsx'
etd = pd.read_excel(file_path)

# Step 2. Initial Data Inspection
print("Initial shape:", etd.shape)
print("\nMissing values:\n", etd.isna().sum())

# Step 3. Feature Selection
exclude_columns = [
    'TRIAL_INDEX',       # Often contains trial identifiers (non-predictive)
    'IA_ID',             # Item/area identifiers (categorical)
    'RECORDING_SESSION_LABEL',    # Participant codes (e.g., 'Sub01')
    'TRIAL_LABEL', # Label identifier for specific sentence
    'IA_LABEL', # Specific Word/Character
]
features = etd.drop(columns=exclude_columns, errors='ignore')

print("After Excluding Columns:", features.columns.tolist())

print(features['LANGUAGE'].unique())

# Step 4: Create Code-Switching Labels
# 0 = non-switch; 1 = C to E / pre-switch; 2: C to E / post-switch; 3 = E to C / pre-switch; 4 = E to C / post-switch
# Initialize all labels to 0 (non-switch)
features['Switch Label'] = 0

# Map conditions to label groups
condition_groups = {
    ('condition 3', 'condition 5', 'condition 7'): {
        'C': 1,  # Pre-switch (C → E)
        'E': 2   # Post-switch (C → E)
    },
    ('condition 4', 'condition 6', 'condition 8'): {
        'E': 3,  # Pre-switch (E → C)
        'C': 4   # Post-switch (E → C)
    }
}

# Apply label mapping
for condition, lang_map in condition_groups.items():
    mask = features['CONDITION'].isin(condition)
    for lang, label in lang_map.items():
        features.loc[mask & (features['LANGUAGE'] == lang), 'Switch Label'] = label

features['Switch Label'] = features['Switch Label'].astype(int)
print("Columns after Adding Switch Label:", features.columns.tolist())

print("Unique classes:", np.unique(features['Switch Label']))
print("Switch Label distribution:", features['Switch Label'].value_counts())

# Step 5: Split Numeric & Categorical Features
categorical_cols = ['L2 PROFICIENCY', 'CONDITION', 'LANGUAGE']
features[categorical_cols] = features[categorical_cols].astype('category')
numeric_cols = [
    'IA_FIRST_FIXATION_DURATION',
    'IA_FIRST_RUN_DWELL_TIME',
    'IA_REGRESSION_PATH_DURATION',
    'IA_DWELL_TIME',
    'IA_FIRST_SACCADE_AMPLITUDE',
    'IA_FIXATION_COUNT',
    'IA_SKIP',
    'IA_FIRST_RUN_FIXATION_COUNT',
    'IA_REGRESSION_IN_COUNT',
    'Switch Label'
]

# Step 6. Missing Value Handling (numeric)
# Strategy: Remove high-missing columns first
missing_pct = features[numeric_cols].isna().mean() * 100
high_missing = missing_pct[missing_pct > 30].index
features = features.drop(columns=high_missing)

print("After Removal of High Missing:", features.columns.tolist())

# Remove critical missing rows
critical_cols = [
    'IA_REGRESSION_PATH_DURATION',
    'IA_FIRST_FIXATION_DURATION',
    'IA_FIRST_RUN_DWELL_TIME',
    'IA_FIRST_SACCADE_AMPLITUDE'
]
features = features.dropna(subset=critical_cols)
print("After Removal of Critical Columns:", features.columns.tolist())

# Step 7. Outlier Detection
z_scores = np.abs(stats.zscore(features[numeric_cols]))
outliers_mask = (z_scores > 3).any(axis=1)
print(f"Found {outliers_mask.sum()} potential outliers")

clean_indices = features[numeric_cols][~outliers_mask].index
features_numeric_cleaned = features.loc[clean_indices, numeric_cols]
features_categorical_cleaned = features.loc[clean_indices, categorical_cols]

features_cleaned = pd.concat([features_numeric_cleaned, features_categorical_cleaned], axis=1)

# Step 11: Define PCA Features of Note
pca_features = [
    'IA_FIRST_FIXATION_DURATION',
    'IA_FIRST_RUN_DWELL_TIME',
    'IA_REGRESSION_PATH_DURATION',
    'IA_DWELL_TIME',
    'IA_FIRST_SACCADE_AMPLITUDE',
    'IA_FIXATION_COUNT',
    'IA_SKIP',
    'IA_FIRST_RUN_FIXATION_COUNT',
    'IA_REGRESSION_IN_COUNT'
]

linguistic_features = ['L2 PROFICIENCY', 'CONDITION', 'LANGUAGE']

# Step 8: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_cleaned[pca_features])

# Step 9. Principal Component Analysis
pca = PCA(n_components=0.95)  # Automatically select components to explain 95% variance
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
    index=pca_features
)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--', color='teal')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

# Visualize Loadings as Heatmap
plt.figure(figsize=(8, 6))
sb.heatmap(loadings, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('PCA Loadings (Feature Contributions)')
plt.tight_layout()
plt.show()

# Output results
print("Explained Variance Ratio:", explained_variance)
print("\nLoadings:\n", loadings)