import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def apply_pca(data, numeric_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[numeric_cols])
    pca = PCA(n_components=0.95)
    components = pca.fit_transform(scaled)

    explained_variance = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(components.shape[1])],
        index=numeric_cols
    )
    return components, explained_variance, loadings