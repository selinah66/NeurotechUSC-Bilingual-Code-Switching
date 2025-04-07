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

# Step 2. Target Separation, separate target FIRST to prevent data leakage
target = etd['Switch Label'] if 'Switch Label' in etd else None
features = etd.drop(columns=['Switch Label'], errors='ignore')

# Step 3. Initial Data Inspection
print("Initial shape:", features.shape)
print("\nMissing values:\n", features.isna().sum())

# Step 4. Numeric Feature Selection
exclude_columns = [
    'TRIAL_INDEX',       # Often contains trial identifiers (non-predictive)
    'IA_ID',             # Item/area identifiers (categorical)
    'RECORDING_SESSION_LABEL',    # Participant codes (e.g., 'Sub01')
    'TRIAL_LABEL', # Label identifier for specific sentence
    'IA_LABEL', # Specific Word/Character
]
features = features.drop(columns=exclude_columns, errors='ignore')

# Perform one-hot encoding for categorical columns
categorical_cols = ['L2 PROFICIENCY', 'CONDITION', 'LANGUAGE']
features = pd.get_dummies(features, columns=categorical_cols)

print(features.dtypes)

# Step 5. Missing Value Handling
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

# Step 7. Feature Scaling
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features_imputed), columns=features_imputed.columns)

# Step 8. Outlier Detection
from scipy import stats
z_scores = np.abs(stats.zscore(features_scaled))
outliers_mask = (z_scores > 3).any(axis=1)
print(f"Found {outliers_mask.sum()} potential outliers")

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

# Filter the features DataFrame to include only the 10 IA features
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