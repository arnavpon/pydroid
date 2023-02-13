import numpy as np
from sklearn.decomposition import PCA


def standardize_mat(mat):
    """
    Make the mean of the matrix 0 and the standard deviation 1.
    """
    mean = np.mean(mat)
    std = np.std(mat)
    centered = (mat - mean) / std 
    return centered

def channel_pca(M, cid, N = 5):

    m = standardize_mat(M)
    pca = PCA(n_components = N)
    pca.fit(m)
    
    return {
        f'{cid}_{i}': np.mean(pca.components_[i, :])
        for i in range(pca.components_.shape[0])
    }
