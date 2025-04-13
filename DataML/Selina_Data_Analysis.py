import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1. Load the dataset
file_path = '/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching/EyeMovementData/IA_data.xlsx'
etd = pd.read_excel(file_path)

# Step 2. Initial Data Inspection
print("Initial shape:", etd.shape)
print("\nMissing values:\n", etd.isna().sum())

# Step 3. Numeric Feature Selection
exclude_columns = [
    'TRIAL_INDEX',       # Often contains trial identifiers (non-predictive)
    'IA_ID',             # Item/area identifiers (categorical)
    'RECORDING_SESSION_LABEL',    # Participant codes (e.g., 'Sub01')
    'TRIAL_LABEL', # Label identifier for specific sentence
    'IA_LABEL', # Specific Word/Character
]
features = etd.drop(columns=exclude_columns, errors='ignore')

# Step 4: Create Code-Switching Labels
# 0 = non-switch; 1 = C to E / pre-switch; 2: C to E / post-switch; 3 = E to C / pre-switch; 4 = E to C / post-switch
# Initialize all labels to 0 (non-switch)
features['Switch Label'] = 0

# Map conditions to label groups
condition_groups = {
    # Chinese-to-English switch conditions
    (3, 5, 7): {
        'C': 1,  # Pre-switch (C → E)
        'E': 2   # Post-switch (C → E)
    },
    # English-to-Chinese switch conditions
    (4, 6, 8): {
        'E': 3,  # Pre-switch (E → C)
        'C': 4   # Post-switch (E → C)
    }
}

# Apply label mapping
for condition_nums, lang_labels in condition_groups.items():
    for num in condition_nums:
        col_name = f'condition {num}'
        if col_name in features.columns:
            # For Chinese language ('C')
            mask_c = (features[col_name] == 1) & (features['LANGUAGE_C'] == 1)
            features.loc[mask_c, 'Switch Label'] = lang_labels['C']

            # For English language ('E')
            mask_e = (features[col_name] == 1) & (features['LANGUAGE_E'] == 1)
            features.loc[mask_e, 'Switch Label'] = lang_labels['E']

features['Switch Label'] = features['Switch Label'].astype(int)
print("Columns in features_imputed:", features.columns.tolist())

# Step 5: Perform one-hot encoding for categorical columns
categorical_cols = ['L2 PROFICIENCY', 'CONDITION', 'LANGUAGE']
features = pd.get_dummies(features, columns=categorical_cols)

# Step 6. Missing Value Handling
# Strategy: Remove high-missing columns first
missing_pct = features.isna().mean() * 100
high_missing = missing_pct[missing_pct > 30].index
features = features.drop(columns=high_missing)

# Remove critical missing rows
critical_cols = [
    'IA_REGRESSION_PATH_DURATION',
    'IA_FIRST_FIXATION_DURATION',
    'IA_FIRST_RUN_DWELL_TIME',
    'IA_FIRST_SACCADE_AMPLITUDE'
]
features = features.dropna(subset=critical_cols)

# Impute remaining missing values
imputer = SimpleImputer(strategy='mean')
features_imputed = pd.DataFrame(imputer.fit_transform(features),
                               columns=features.columns)

# Step 6. Imputation of Features Visualization (Before/After)
plt.figure(figsize=(10, 6))
sb.kdeplot(etd['IA_DWELL_TIME'], label='Original', color='blue')
sb.kdeplot(features_imputed['IA_DWELL_TIME'], label='Cleaned', color='green')
plt.title('Data Distribution Before/After Cleaning')
plt.xlabel('IA_DWELL_TIME')
plt.ylabel('Density')
plt.legend()
plt.show()

# Step 8. Outlier Detection
from scipy import stats
z_scores = np.abs(stats.zscore(features_imputed))
outliers_mask = (z_scores > 3).any(axis=1)
print(f"Found {outliers_mask.sum()} potential outliers")
features_cleaned = features_imputed[~outliers_mask]

print(features_imputed.shape)
print(features_cleaned.shape)


# Filter the features DataFrame to include only the 10 IA features
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
features_filtered = features_scaled[pca_features]

# Step 9. Principal Component Analysis
pca = PCA(n_components=0.95)  # Automatically select components to explain 95% variance
features_pca = pca.fit_transform(features_filtered)
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

# Step 10. Initial Data Visualization