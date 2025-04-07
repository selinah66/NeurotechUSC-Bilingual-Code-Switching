import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

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
categorical_cols = ['L2 PROFICIENCY', 'LANGUAGE']
features = pd.get_dummies(features, columns=categorical_cols)
features = pd.get_dummies(features, columns=['CONDITION'], prefix='', prefix_sep='')

print(features.dtypes)
print("Columns after encoding:", features.columns.tolist())

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

# Step 7. Outlier Detection
from scipy import stats
z_scores = np.abs(stats.zscore(features_imputed))
outliers_mask = (z_scores > 3).any(axis=1)
print(f"Found {outliers_mask.sum()} potential outliers")

print("LANGUAGE_C values:", features_imputed['LANGUAGE_C'].unique())
print("LANGUAGE_E values:", features_imputed['LANGUAGE_E'].unique())

# Step 8: Create Code-Switching Labels
# 0 = non-switch; 1 = C to E / pre-switch; 2: C to E / post-switch; 3 = E to C / pre-switch; 4 = E to C / post-switch
# Initialize all labels to 0 (non-switch)
features_imputed['Switch Label'] = 0

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
        if col_name in features_imputed.columns:
            # For Chinese language ('C')
            mask_c = (features_imputed[col_name] == 1) & (features_imputed['LANGUAGE_C'] == 1)
            features_imputed.loc[mask_c, 'Switch Label'] = lang_labels['C']

            # For English language ('E')
            mask_e = (features_imputed[col_name] == 1) & (features_imputed['LANGUAGE_E'] == 1)
            features_imputed.loc[mask_e, 'Switch Label'] = lang_labels['E']

features_imputed['Switch Label'] = features_imputed['Switch Label'].astype(int)
print("Columns in features_imputed:", features_imputed.columns.tolist())

# Verify labels
print("Unique classes:", np.unique(features_imputed['Switch Label']))
print("Class counts:", np.bincount(features_imputed['Switch Label'].astype(int)))

# Define PCA Features of Note
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
features_filtered = features_imputed[pca_features]

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

# Separate labels from features BEFORE scaling
label_features = features_imputed['Switch Label'].copy()
features_to_scale = features_imputed.drop(columns=['Switch Label'])

# Step 10. Feature Scaling
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features_to_scale), columns=features_to_scale.columns)

features_scaled['Switch Label'] = label_features.values

# X_pca = features_scaled[:, :4]

# Prepare features and labels
CS_features = features_scaled[pca_features]
label_feature = features_scaled['Switch Label']

X_pca = features_pca[:, :3]

# Step 11: Create feature vectors for each word
X = CS_features.values
y = label_feature.values

# Step 12: Training the SVM Model
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y)

# Verify class distribution
print("Unique classes in training:", np.unique(y_train))
if len(np.unique(y_train)) < 2:
    print("ERROR: Only one class in training data!")

# Verify class distribution in the training and test sets
print("# of rows:", features_scaled.shape[0])
print("# of columns:", features_scaled.shape[1])
print("Class distribution in the full dataset:", Counter(y))
print("Class distribution in the training set:", Counter(y_train))
print("Class distribution in the test set:", Counter(y_test))
print("NaN values in X_train:", np.isnan(X_train).sum())
print("NaN values in X_train:", np.isnan(X_test).sum())

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_pca, y)
print("Best parameters:", grid.best_params_)

# Train SVM model
print("\n=== Support Vector Machine ===")
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight={i: w for i, w in enumerate(class_weights)})
svm.fit(X_train, y_train)

# Evaluate SVM
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')

print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"SVM F1 Score: {svm_f1:.4f}")
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
