import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config_lang import RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_LEAF, MIN_SAMPLES_SPLIT, N_SPLITS, N_REPEATS, MAX_FEATURES

# Train random forest model
def train_random_forest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
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


# Tune random forest model using RandomizedSearchCV to optimize hyperparameters and test CV configurations
def tune_random_forest(X, y):
    model = RandomForestClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [4, 8],
        'max_features': ['sqrt', 0.3, 0.5],
        'bootstrap': [True],
        'max_leaf_nodes': [50, 100]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        cv=RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit='f1'
    )

    search.fit(X, y)

    print("\nBest Overfitting-Reducing Configuration")
    print("Key parameters for generalization:")
    print(f"Best max_depth: {search.best_params_['max_depth']}")
    print(f"Best min_samples_split: {search.best_params_['min_samples_split']}")
    print(f"Best min_samples_leaf: {search.best_params_['min_samples_leaf']}")
    print(f"Best max_features: {search.best_params_['max_features']}")
    print(f"\nBest F1 Score (CV): {search.best_score_:.2f}")

    return search.best_estimator_

# Evaluate model performance using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)
    metrics = ['accuracy', 'f1_weighted']

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=metrics,
        return_train_score=True,
        n_jobs=-1
    )

    print("\nOverfitting Analysis")
    print(f"Train Accuracy: {results['train_accuracy'].mean():.2f} (±{results['train_accuracy'].std():.2f})")
    print(f"Test Accuracy:  {results['test_accuracy'].mean():.2f} (±{results['test_accuracy'].std():.2f})")
    print(f"\nTrain F1: {results['train_f1_weighted'].mean():.2f} (±{results['train_f1_weighted'].std():.2f})")
    print(f"Test F1:  {results['test_f1_weighted'].mean():.2f} (±{results['test_f1_weighted'].std():.2f})")

    return results

