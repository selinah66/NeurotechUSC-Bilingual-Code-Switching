import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def apply_pca(data, numeric_cols, selected_features=None):
    # Scale numeric columns
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[numeric_cols])

    # Apply PCA to retain 95% variance
    pca = PCA(n_components=0.95)
    components = pca.fit_transform(scaled)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # PCA Loadings (importance of each original feature per PC)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i + 1}' for i in range(components.shape[1])],
        index=numeric_cols
    )

    # ===== Visualizations =====

    # 1. Scree plot (explained variance)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', color='navy')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% Threshold')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. PCA Loadings bar plot for PC1 (or top PCs)
    top_pc = 'PC1'
    plt.figure(figsize=(10, 5))
    loadings[top_pc].sort_values(key=abs, ascending=False).plot(kind='bar', color='teal')
    plt.title(f'Feature Loadings for {top_pc}')
    plt.ylabel('Loading Magnitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Highlighting selected features
    if selected_features:
        print("\nTop contributing features in PCA (based on loading magnitude):")
        top_features = loadings[top_pc].abs().sort_values(ascending=False)
        for feat in selected_features:
            print(f"  {feat}: loading = {loadings.loc[feat, top_pc]:.3f}")

    return components, explained_variance, loadings
