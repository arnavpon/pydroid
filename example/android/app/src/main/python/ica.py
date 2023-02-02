import numpy as np
np.random.seed(0)

def vanilla_ica(X, iterations = 1000, tolerance = 1e-5):
    """
    https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e
    """
    
    # center and whiten component matrix
    components = _center(X)
    components = _whiten(components)
    components_nr = X.shape[0]
    
    W = np.zeros((components_nr, components_nr), dtype = X.dtype)
    for i in range(components_nr):
        
        w = np.random.rand(components_nr)
        for _ in range(iterations):
            
            w_new = _calculate_new_w(w, X)
            if i >= 1:
                w_new -= np.dot(
                    np.dot(w_new, W[:i].T),
                    W[:i]
                )
            
            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            w = w_new
            if distance < tolerance:
                break
                
        W[i, :] = w
            
    S = np.dot(W, X)
    return S

def _g(x):
    return np.tanh(x)

def _g_der(x):
    return 1 - _g(x) * _g(x)

def _center(X):
    X = np.array(X)
    mean = X.mean(axis = 1, keepdims = True)
    return X - mean

def _whiten(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)

    inv = np.linalg.inv(D)
    eigenvalues, eigenvectors = np.linalg.eig(inv)
    D_inv = np.dot(eigenvectors, np.dot(np.diag(np.sqrt(eigenvalues)), np.linalg.inv(eigenvectors)))


    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten

def _calculate_new_w(w, X):
    w_new = (X * _g(np.dot(w.T, X))).mean(axis=1) - _g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new


def jade(X):

    m, n = X.shape
    W = np.eye(m)

    for i in range(n - 1):
        for j in range(i + 1, n):
            
            # compute the Givens rotation
            x1, x2 = X[:, i], X[:, j]
            G = np.eye(m)
            d = np.sqrt(x1.dot(x1) + x2.dot(x2))
            c = x1.dot(x2) / d
            
            c_res = 1 - c**2
            s = np.sqrt(c_res) if c_res > 0 else 0
            
            G[i, i], G[j, j] = c, c
            G[i, j] = s
            G[j, i] = -s

            # update the estimated eigenmatrix
            X = G.dot(X)
            W = W.dot(G.T)

    return W.dot(X)
