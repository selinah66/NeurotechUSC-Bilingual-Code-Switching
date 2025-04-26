import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config_lang import (
    CV_FOLDS, MAX_DEPTH, MAX_FEATURES, MIN_SAMPLES_LEAF,
    MIN_SAMPLES_SPLIT, N_ESTIMATORS, RANDOM_STATE
)

# Compare performance of different classifiers using cross-validation
def compare_classifiers(X, y, cv=CV_FOLDS):
    """Compare performance of different classifiers using cross-validation."""
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            random_state=RANDOM_STATE
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(random_state=RANDOM_STATE),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=RANDOM_STATE)
    }
    
    results = {
        name: {
            'mean_accuracy': cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean(),
            'std_accuracy': cross_val_score(clf, X, y, cv=cv, scoring='accuracy').std()
        }
        for name, clf in classifiers.items()
    }
    
    # Plot results
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    means = [results[name]['mean_accuracy'] for name in names]
    stds = [results[name]['std_accuracy'] for name in names]
    
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Mean Accuracy (Cross-Validation)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('classifier_comparison.png')
    plt.show()
    
    return results

# Analyze Random Forest stability across multiple runs
def analyze_model_stability(X, y, n_iterations=10, cv=CV_FOLDS):
    """Analyze Random Forest stability across multiple runs."""
    accuracies = []
    feature_importances = []
    
    for i in range(n_iterations):
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            random_state=RANDOM_STATE + i
        )
        accuracies.append(cross_val_score(rf, X, y, cv=cv).mean())
        rf.fit(X, y)
        feature_importances.append(rf.feature_importances_)
    
    # Print and plot results
    print("\nModel Stability Analysis:")
    print(f"Mean Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    
    stability = np.std(feature_importances, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(X.columns, stability)
    plt.title('Feature Importance Stability')
    plt.xlabel('Features')
    plt.ylabel('Standard Deviation of Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_stability.png')
    plt.show()

if __name__ == "__main__":
    # Import preprocessed data from main.py
    from main import X, y
    
    # Run experiments
    print("Classifier Comparison")
    compare_classifiers(X, y)
    
    print("\nModel Stability Analysis")
    analyze_model_stability(X, y)
