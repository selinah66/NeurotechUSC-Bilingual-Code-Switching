# model_experiments.py

import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Assume X and y are already preprocessed and passed in
def evaluate_models(X, y, top_features=None):
    if top_features:
        X = X[top_features]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n[Model Experimentation]")
    print("Class distribution in the train set:", Counter(y_train))
    print("Class distribution in the test set:", Counter(y_test))

    ### Gradient Boosting
    print("\n--- Gradient Boosting ---")
    param_grid_gb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    gb = GradientBoostingClassifier(random_state=42)
    grid_search_gb = GridSearchCV(gb, param_grid=param_grid_gb, cv=3, n_jobs=-1, verbose=0)
    grid_search_gb.fit(X_train, y_train)
    gb_best = grid_search_gb.best_estimator_
    gb_preds = gb_best.predict(X_test)

    print("Best Gradient Boosting params:", grid_search_gb.best_params_)
    print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_preds))
    print(classification_report(y_test, gb_preds))

    ### Logistic Regression
    print("\n--- Logistic Regression ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    log_preds = log_reg.predict(X_test_scaled)

    print("Logistic Regression Accuracy:", accuracy_score(y_test, log_preds))
    print(classification_report(y_test, log_preds))

    ### Support Vector Machine
    print("\n--- Support Vector Machine (SVM) ---")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_preds = svm.predict(X_test_scaled)

    print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
    print(classification_report(y_test, svm_preds))
