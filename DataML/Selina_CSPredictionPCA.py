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
from scipy.stats.mstats import winsorize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler

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

# Step 7. Outlier Detection
from scipy import stats
z_scores = np.abs(stats.zscore(features_imputed))
outliers_mask = (z_scores > 3).any(axis=1)
print(f"Found {outliers_mask.sum()} potential outliers")
features_cleaned = features_imputed[~outliers_mask]

print(features_imputed.shape)
print(features_cleaned.shape)

# Step 11: Create feature vectors for each word
label_features = features_imputed['Switch Label'].copy()
features_to_scale = features_imputed.drop(columns=['Switch Label'])
label_feature = features_imputed['Switch Label']

X = features_imputed.drop(columns=['Switch Label'])
y = label_feature

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

feature_columns = X_train.columns.tolist()

# Step 8. Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=feature_columns)

# Verify labels
print("Unique classes:", np.unique(features_imputed['Switch Label'])) ## << error
print("Class counts:", np.bincount(features_imputed['Switch Label'].astype(int)))

# Export to a New Cleaned Data (remember to ## when running or will create many new files)
file_path = 'cleaned_data.xlsx'
features_cleaned.to_excel(file_path, index=False)  # Export without row indices
print(f"Cleaned data has been saved to {file_path}")



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





# Step 12: Training the SVM Model

# Verify class distribution
print("Unique classes in training:", np.unique(y_train))
if len(np.unique(y_train)) < 2:
    print("ERROR: Only one class in training data!")

# Verify class distribution in the training and test sets
print("Shape of X & y:", features_scaled.shape)
print("Class distribution in the full dataset:", Counter(y))
print("Class distribution in the training set:", Counter(y_train))
print("Class distribution in the test set:", Counter(y_test))

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

#grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
#grid.fit(X_pca, y)
#print("Best parameters:", grid.best_params_)

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
