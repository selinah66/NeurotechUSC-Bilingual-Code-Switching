import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split, GroupKFold, GroupShuffleSplit, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

skipGraphs = False

def doGraphs(low_prof, high_prof):
	if skipGraphs:
		return
		
	# Step 2: Initial Data Visualization
	# Plot 1. Scatter Plot: Fixation Count vs. First Saccade Amplitude by L2 Proficiency
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
	
	# Plot for L2 = 'L'
	ax1.scatter(low_prof.iloc[:, 13], low_prof.iloc[:, 12],
			   alpha=0.5, edgecolors='w', linewidth=0.5, c='blue')
	ax1.set_title("Fixation Count vs. Saccade Amplitude with L2 Proficiency = L")
	ax1.set_xlabel("Fixation Count")
	ax1.set_ylabel("Saccade Amplitude")
	
	# Plot for L2 = 'H'
	ax2.scatter(high_prof.iloc[:, 13], high_prof.iloc[:, 12],
			   alpha=0.5, edgecolors='w', linewidth=0.5, c='red')
	ax2.set_title("Fixation Count vs. Saccade Amplitude with L2 Proficiency = H")
	ax2.set_xlabel("Fixation Count")
	
	# Adjust layout and show plot
	plt.tight_layout()
	plt.show()
	
	#-------------------------------------------------------------------	
	
	# Plot 2. Bar Chart: Average fixation durations
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
	
	# Plot bar chart
	avg_first_fixation = pd.to_numeric(low_prof.iloc[:, 7], errors='coerce').mean()
	avg_second_fixation = pd.to_numeric(low_prof.iloc[:, 8], errors='coerce').mean()
	ax1.bar(['First Fixation', 'Second Fixation'], [avg_first_fixation, avg_second_fixation])
	ax1.set_xlabel('Fixation Type')
	ax1.set_ylabel('Average Duration (s)')
	ax1.set_title('Average Fixation Durations with L2 Proficiency = L')
	
	avg_first_fixation = pd.to_numeric(high_prof.iloc[:, 7], errors='coerce').mean()  # Assuming Column 8 is the first fixation duration
	avg_second_fixation = pd.to_numeric(high_prof.iloc[:, 8], errors='coerce').mean()
	ax2.bar(['First Fixation', 'Second Fixation'], [avg_first_fixation, avg_second_fixation])
	ax2.set_xlabel('Fixation Type')
	ax2.set_ylabel('Average Duration (s)')
	ax2.set_title('Average Fixation Durations with L2 Proficiency = H')
	
	plt.tight_layout()
	plt.show()

	#-------------------------------------------------------------------	

	# Plot 3. Stacked Bar Chart: Regression Paths vs. First Run Dwell Time
	# Calculate sums for each type of gaze behavior
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
	
	regression_path_time = low_prof.iloc[:, 10].sum()  # Assuming Column 11 is Regression Path Duration
	first_run_dwell_time = low_prof.iloc[:, 9].sum()  # Assuming Column 10 is First Run Dwell Time
	ax1.bar(['Regression Paths', 'First Run Dwell Time'], [regression_path_time, first_run_dwell_time])
	ax1.set_xlabel('Gaze Behavior')
	ax1.set_ylabel('Total Time (s)')
	ax1.set_title('Gaze Behavior Comparison with L2 Proficiency = L')
	
	regression_path_time = high_prof.iloc[:, 10].sum()  # Assuming Column 11 is Regression Path Duration
	first_run_dwell_time = high_prof.iloc[:, 9].sum()  # Assuming Column 10 is First Run Dwell Time
	ax2.bar(['Regression Paths', 'First Run Dwell Time'], [regression_path_time, first_run_dwell_time])
	ax2.set_xlabel('Gaze Behavior')
	ax2.set_ylabel('Total Time (s)')
	ax2.set_title('Gaze Behavior Comparison with L2 Proficiency = H')
	
	plt.tight_layout()
	plt.show()

def plotIA_DWELL_TIME(etd, etd_cleaned, etd_numeric_imputed):
	if skipGraphs:
		return
		
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

def doMorePlots(etd, etd_cleaned):
	if skipGraphs:
		return
	import matplotlib.pyplot as plt
	import seaborn as sns
	
	# Box plot to visualize outliers
	plt.figure(figsize=(12, 6))
	sns.boxplot(data=etd_cleaned[['IA_REGRESSION_PATH_DURATION']])
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

# Step 0: Load the dataset
etd = pd.read_excel('/Users/Selina/Documents/GitHub/NeurotechUSC-Bilingual-Code-Switching/Data/RawEyeMovement.xlsx')
print("Total Rows=", etd.shape[0])

# Step 1: Remove non-English Rows
# Keep only rows where "IA_LABEL" starts with an alphabet character (A-Z or a-z)
etd = etd[etd['IA_LABEL'].str.match(r'^[a-zA-Z]', na=False)]

# Display the filtered DataFrame
print("After removing non-English rows.  Total Rows=", etd.shape[0])
low_prof = etd[etd['L2 PROFICIENCY'] == 'L']
high_prof = etd[etd['L2 PROFICIENCY'] == 'H']

doGraphs(low_prof, high_prof)

# Step 3: Data Cleaning & Quality Check
# Check for missing values
print("===========> Cleaning Data <=============")
print("Check for missing values:\n", etd.isnull().sum())

# Percentage of missing values
missing_percentage = (etd.isnull().sum() / len(etd)) * 100
print(missing_percentage)

# Drop columns with >30% missing values (Cleaning)
missing_threshold = 30
columns_to_drop = missing_percentage[missing_percentage > missing_threshold].index
print("Columns dropped:", list(columns_to_drop))  # Print the columns being dropped
etd_cleaned = etd.drop(columns=columns_to_drop)

# Remove rows with empty 'IA_REGRESSION_PATH_DURATION','IA_FIRST_FIXATION_DURATION' & 'IA_FIRST_RUN_DWELL_TIME' values.
columns_to_check = ['IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME']
etd_cleaned = etd_cleaned[~(etd_cleaned[columns_to_check].isnull().all(axis=1))]

# Remove more rows for which they have empty 'IA_FIRST_SACCADE_AMPLITUDE'
etd_cleaned = etd_cleaned[~(etd_cleaned[['IA_FIRST_SACCADE_AMPLITUDE']].isnull().all(axis=1))]
print("Check for missing values:\n", etd_cleaned.isnull().sum())

# Impute remaining numeric columns
numeric_cols = etd_cleaned.select_dtypes(include=['int64', 'float64']).columns
numeric_imputer = SimpleImputer(strategy='mean')
etd_numeric_imputed = etd_cleaned.copy()
etd_numeric_imputed[numeric_cols] = numeric_imputer.fit_transform(etd_cleaned[numeric_cols])
print("===========> Imputed data for remaining numeric columns <=============")

#================ PLOT IA_DWELL_TIME graph
plotIA_DWELL_TIME(etd, etd_cleaned, etd_numeric_imputed)

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
print("====> Removing outliers <======")
etd_cleaned = remove_outliers(etd_numeric_imputed, 'IA_REGRESSION_PATH_DURATION', method='iqr')
print(f"Original data: {len(etd)}, After removing outliers: {len(etd_cleaned)}")

#================ MORE PLOTS
doMorePlots(etd, etd_cleaned)

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

#feature_columns = ['IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 
#                  'IA_REGRESSION_PATH_DURATION', 'IA_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE']
feature_columns = ['IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION', 'IA_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE']

#==============================================================================
# Data Aggregation per Session for ["mean", "median", "std", "min", "max"] of feature columns
etd_data = etd_cleaned.copy()

# Aggregate features per session
agg_funcs = ["mean", "median", "std", "min", "max"]
grouped = etd_data.groupby("RECORDING_SESSION_LABEL")[feature_columns].agg(agg_funcs)
grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]  # Flatten column names

# Merge with target (L2 Proficiency)
target = etd_data.drop_duplicates("RECORDING_SESSION_LABEL").set_index("RECORDING_SESSION_LABEL")["L2 PROFICIENCY"]
data = grouped.join(target).dropna()

# Split into features (X) and target (y)
X = data.drop("L2 PROFICIENCY", axis=1)
y = data["L2 PROFICIENCY"]

# Stratified Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, 
    stratify=y,  # Preserve L/H ratio in splits
    random_state=42
)

# Initialize classifier (e.g., Random Forest)
clf = RandomForestClassifier(n_estimators=1000, max_depth=200, random_state=42)

# Option 1: Train on training data and evaluate on test data
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"========================================================")
print(f"Data Aggregation per Session for [mean, median, std, min, max] of feature columns")
print(f"Using RandomForestClassifier(n_estimators=1000, max_depth=200, random_state=42)\n")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))  # Precision/Recall/F1

# Option 2: Cross-validate on training data (recommended for robustness)
cv = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
print(f"Mean CV Accuracy (Train): {cv_scores.mean():.2f}")

# Final step: Train on full training data and evaluate on held-out test data
clf.fit(X_train, y_train)
final_test_accuracy = clf.score(X_test, y_test)
print(f"Final Test Accuracy: {final_test_accuracy:.2f}")
print(f"========================================================")
#==============================================================================
'''
param_grid = {
    'n_estimators': [200, 500, 1000, 2000],
    'max_depth': [10, 200, 500, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1]
}

#rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
predictions = best_rf.predict(X_test)

print(f"\n\n\n========================================================")
print(f"Use GridSearchCV to find Best Parameters.")
print("Best parameters found: ", grid_search.best_params_)
print("Accuracy with best parameters: ", accuracy_score(y_test, predictions))
print(f"========================================================")
#==============================================================================
'''
# Hyperparameter Tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

gb = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=3, n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train, y_train)
best_gb = grid_search_gb.best_estimator_
gb_predictions = best_gb.predict(X_test)
print(f"\n\n\n========================================================")
print("Best parameters for Gradient Boosting found: ", grid_search_gb.best_params_)
print("Gradient Boosting accuracy with best parameters: ", accuracy_score(y_test, gb_predictions))
accuracy_score(y_test, gb_predictions)

# Trying Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_reg_predictions = log_reg.predict(X_test_scaled)
print("Logistic Regression accuracy: ", accuracy_score(y_test, log_reg_predictions))
print(f"=============================================")

#==============================================================================
# Pivot the rows to become columns
print(f"===========> Pivot the rows to become columns <===========")

data = etd_cleaned.copy()

# Create pivoted data using pivot_table directly
pivoted_features = data.pivot_table(
    index='RECORDING_SESSION_LABEL',
    columns='IA_LABEL',
    values=feature_columns,
    aggfunc='mean'
)

# Flatten multi-index columns format "feature_IA_LABEL"
pivoted_features.columns = [f"{feature}_{label}" for feature, label in pivoted_features.columns]

# Reset index to make "RECORDING_SESSION_LABEL" a column again
pivoted_features.reset_index(inplace=True)

# Merge with target variable which uniquely maps to `RECORDING_SESSION_LABEL`
unique_records = data.drop_duplicates('RECORDING_SESSION_LABEL')[['RECORDING_SESSION_LABEL', 'L2 PROFICIENCY']]
pivoted_data = unique_records.merge(pivoted_features, on='RECORDING_SESSION_LABEL')

# First, check the distribution of NaN values in the pivoted_data DataFrame (excluding the first two columns)
nan_distribution = pivoted_data.drop(columns=['RECORDING_SESSION_LABEL', 'L2 PROFICIENCY']).isnull().sum()
# print(nan_distribution[nan_distribution > 0])

# Define a threshold for the maximum allowed NaN values in a column (e.g., 50% of the rows)
# i.3. remove columns that have # of NaN rows > 50%
threshold = 0.5 * len(pivoted_data)

# Identify columns where the number of NaN values exceeds the threshold
columns_to_drop = nan_distribution[nan_distribution > threshold].index

# Drop these columns from the pivoted_data DataFrame
pivoted_data_cleaned = pivoted_data.drop(columns=columns_to_drop)

# Fill remaining NaN values with the mean of the column
'''
imputer = SimpleImputer(strategy='mean')
pivoted_data_imputed = pd.DataFrame(imputer.fit_transform(pivoted_data_cleaned.drop(columns=['RECORDING_SESSION_LABEL', 'L2 PROFICIENCY'])), 
                                    columns=pivoted_data_cleaned.drop(columns=['RECORDING_SESSION_LABEL', 'L2 PROFICIENCY']).columns)
pivoted_data_imputed['RECORDING_SESSION_LABEL'] = pivoted_data_cleaned['RECORDING_SESSION_LABEL']
pivoted_data_imputed['L2 PROFICIENCY'] = pivoted_data_cleaned['L2 PROFICIENCY']

pivoted_data_cleaned = pivoted_data_imputed.copy()
'''

# Separate features from the target variable
X = pivoted_data_cleaned.drop(columns=['RECORDING_SESSION_LABEL', 'L2 PROFICIENCY'])
y = pivoted_data_cleaned['L2 PROFICIENCY']

# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verify class distribution in the training and test sets
print("# of rows:", pivoted_data_cleaned.shape[0])
print("# of columns:", pivoted_data_cleaned.shape[1])

print("Class distribution in the full dataset:", Counter(y))
print("Class distribution in the training set:", Counter(y_train))
print("Class distribution in the test set:", Counter(y_test))
print("NaN values in X_train:", X_train.isnull().sum().sum())
print("NaN values in X_test:", X_test.isnull().sum().sum())

print(f"\n\n\n========================================================")
print(f"Use RandomForestClassifier() on the Pivoted data with (n_estimators=1000, max_depth=200).")
# Train Random Forest
rf = RandomForestClassifier(n_estimators=1000, max_depth=200, random_state=42)
rf.fit(X_train, y_train)

# For validation, you can use:
predictions = rf.predict(X_test)

# Test set evaluation
print("\nTest Set Performance:")
print(classification_report(y_test, predictions))
print(f"Test Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(f"========================================================")

param_grid = {
    'n_estimators': [50, 100, 500, 1000],
    'max_depth': [10, 200, 500, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1]
}

#rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
predictions = best_rf.predict(X_test)

print(f"\n\n\n========================================================")
print(f"Use GridSearchCV to find Best Parameters.")

print("Best parameters found: ", grid_search.best_params_)
print("Accuracy with best parameters: ", accuracy_score(y_test, predictions))
print(f"========================================================")
# Feature importance:
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame(importances, index=feature_names, columns=['importance']).sort_values(by='importance', ascending=False)

print("\n\n\n======> Print feature importances <===========")
print(feature_importances)
print("======> END OF Print feature importances <===========")

#############################
# Cross-validation:
from sklearn.model_selection import cross_val_score

scores = cross_val_score(best_rf, X, y, cv=5)
print("Cross-validation scores: ", scores)
print("Average cross-validation score: ", scores.mean())

'''
# Trying another model such as Gradient Boosting:
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
gb_predictions = gb.predict(X_test)
print("Gradient Boosting accuracy: ", accuracy_score(y_test, gb_predictions))
'''


############################

# Keep top N features (e.g., top 50 features)
top_n = 200
top_features = feature_importances.head(top_n).index
X_top = X[top_features]

# Split into training and validation sets again
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest again
rf = RandomForestClassifier(n_estimators=2000, max_depth=500, random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print(f"\n\n\n========================================================")
print(f"Use RandomForestClassifier after keeping only the top 200 features.")
print("Random Forest accuracy with top {} features: {}".format(top_n, accuracy_score(y_test, predictions)))
#############################
# Cross-validation with top N features
scores = cross_val_score(rf, X_top, y, cv=5)
print("Cross-validation scores with top {} features: {}".format(top_n, scores))
print("Average cross-validation score with top {} features: {}".format(top_n, scores.mean()))
print(f"==============================================================\n\n\n")
# Hyperparameter Tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [10, 50, 100, 200, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1]
}

rf = RandomForestClassifier(n_estimators=2000, max_depth=500, random_state=42)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

best_rf = grid_search_rf.best_estimator_
predictions = best_rf.predict(X_test)
print(f"\n\n\n========================================================")
print(f"Still for dataset with only the top 200 features.")
print("Best parameters for Random Forest found: ", grid_search_rf.best_params_)
print("Random Forest accuracy with best parameters: ", accuracy_score(y_test, predictions))
print(f"==============================================================\n\n\n")
'''
# Hyperparameter Tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

gb = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=3, n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train, y_train)
best_gb = grid_search_gb.best_estimator_
gb_predictions = best_gb.predict(X_test)
print("Best parameters for Gradient Boosting found: ", grid_search_gb.best_params_)
print("Gradient Boosting accuracy with best parameters: ", accuracy_score(y_test, gb_predictions))

# Trying Support Vector Machine (SVM)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
svm_predictions = svm.predict(X_test_scaled)
print("SVM accuracy: ", accuracy_score(y_test, svm_predictions))

# Trying Logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_reg_predictions = log_reg.predict(X_test_scaled)
print("Logistic Regression accuracy: ", accuracy_score(y_test, log_reg_predictions))
'''