import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# --- Step 0: Load the dataset from Sheet1 ---
file_path = '/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching/EyeMovementData/updated_IA_CS.xlsx'
df = pd.read_excel(file_path, sheet_name="Sheet1")

# --- Identify numeric columns (features) ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns detected:", numeric_cols)

# Exclude the target label column ("Switch Label") from features if present.
target_column = "Switch Label"
if target_column in numeric_cols:
    numeric_cols.remove(target_column)
print("\nNumeric feature columns after excluding target:", numeric_cols)

# Select features using the cleaned numeric columns.
features = df[numeric_cols].copy()

# --- Step 1: Check Imputation ---
print("\nMissing values per numeric column:")
print(features.isna().sum())

# Impute missing values using the mean of each column.
imputer = SimpleImputer(strategy="mean")
features_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

# --- Step 2: Z-score Normalization ---
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features_imputed), columns=features_imputed.columns)

# --- Step 3: Principal Component Analysis (PCA) ---
# Retain components that explain 95% of the variance.
pca = PCA(n_components=0.95, random_state=42)
features_pca = pca.fit_transform(features_scaled)

# Plot the cumulative explained variance ratio.
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()

# --- Step 4: Correlation Analysis ---
corr_matrix = features_imputed.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Imputed Features")
plt.show()

print("\nExplained variance ratio per component:")
for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"Component {i}: {ratio:.4f}")

# (Optional) Save the PCA-transformed features for further modeling.
processed_df = pd.DataFrame(features_pca)
processed_df.to_csv('/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching/DataML', index=False)
print("Processed data saved to processed_features.csv")

# --- Step 5: Prepare Data for Modeling ---
# Use the PCA-transformed features (features_pca) as the input matrix.
# Extract the target labels ("Switch Label") from the original DataFrame.
X = features_pca  # PCA-transformed features.
y = df[target_column].values  # Target labels (0, 1, 2).

# Split data into training and testing sets with stratification to maintain class balance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Step 6: Train an SVM Classifier ---
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("=== SVM Performance ===")
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_svm)))
print("Weighted F1 Score: {:.4f}".format(f1_score(y_test, y_pred_svm, average='weighted')))
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# --- (Optional) Step 7: Train a Random Forest Classifier ---
rf = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=70,
                            min_samples_split=10, min_samples_leaf=4)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("=== Random Forest Performance ===")
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_rf)))
print("Weighted F1 Score: {:.4f}".format(f1_score(y_test, y_pred_rf, average='weighted')))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))