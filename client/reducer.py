from sklearn.decomposition import PCA

def reduce_dimensions(X, variance=0.95):
    pca = PCA(n_components=variance)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca
