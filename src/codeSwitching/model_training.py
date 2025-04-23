import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config_switch import RANDOM_STATE, CV_FOLDS, N_ESTIMATORS, MAX_DEPTH
from sklearn.metrics import precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize


def train_random_forest(X_train, y_train, X_test, y_test):
    # Apply SMOTE to balance the classes
    smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Define RandomForestClassifier with adjusted hyperparameters
    clf = RandomForestClassifier(
        n_estimators=100,  # Lower the number of estimators
        max_depth=10,  # Limit tree depth to prevent overfitting
        random_state=RANDOM_STATE,
        class_weight="balanced",  # Account for class imbalance
        min_samples_split=10,  # Increase minimum samples to split
        min_samples_leaf=5,  # Ensure minimum samples in leaves
        max_features='sqrt',  # Limit features considered for each split
        bootstrap=True  # Bootstrapping for better generalization
    )

    # Train the model on resampled data
    clf.fit(X_res, y_res)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate model performance
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Binarize the labels for multiclass precision-recall
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])

    # Get predicted probabilities for each class
    y_test_prob = clf.predict_proba(X_test)

    # Initialize plot
    plt.figure(figsize=(10, 8))

    # Compute and plot precision-recall curve for each class
    for i in range(5):  # For each class (0, 1, 2, 3, 4)
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_test_prob[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'Class {i} (AUC = {pr_auc:.2f})')

    # Plot the precision-recall curve for all classes
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multiclass Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return clf


def cross_validate_model(clf, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    print("CV Accuracy:", scores.mean())


def tune_random_forest(X_train, y_train):
    # Hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [200, 500, 1000],  # Number of trees
        'max_depth': [10, 200, None],  # Maximum depth of each tree
        'min_samples_split': [10, 20, 30],  # Minimum samples required to split a node
        'min_samples_leaf': [5, 10, 20],  # Minimum samples required in a leaf node
    }

    # Initialize the model with default parameters
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    # Set up GridSearchCV for hyperparameter tuning
    grid = GridSearchCV(
        model,
        param_grid,
        cv=CV_FOLDS,
        scoring="balanced_accuracy",  # To deal with class imbalance
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    # Return the best model from the grid search
    print("Best Parameters:", grid.best_params_)
    return grid
