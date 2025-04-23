import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from config_lang import RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_LEAF, MIN_SAMPLES_SPLIT, N_SPLITS, N_REPEATS


# === Core Model Training ===
def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate final model"""
    clf = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features='sqrt',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Model evaluation
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return clf


# === Hyperparameter Tuning ===
def tune_random_forest(X, y):
    """Simplified grid search with cross-validation"""
    model = RandomForestClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [5, 10],
        'max_features': ['sqrt', 0.5]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)
    print(f"Best F1: {grid.best_score_:.2f}")
    return grid.best_estimator_


# === Feature Importance Visualization ===
def plot_feature_importance(clf, feature_names):
    """Plot importance using feature names list"""
    importances = clf.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importances, align="center")
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title("Feature Importances")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.show()

# === Model Evaluation ===
def evaluate_model(model, X, y):
    """Standard cross-validation evaluation"""
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=['accuracy', 'f1_weighted'],
        n_jobs=-1
    )

    print("\n=== Cross-Validation Results ===")
    print(f"Accuracy: {results['test_accuracy'].mean():.2f} (±{results['test_accuracy'].std():.2f})")
    print(f"F1-Score: {results['test_f1_weighted'].mean():.2f} (±{results['test_f1_weighted'].std():.2f})")
    return results
