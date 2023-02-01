import numpy as np
np.random.seed(0)

def vanilla_ica(X, iterations, tolerance = 1e-5):
    
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
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten

def _calculate_new_w(w, X):
    w_new = (X * _g(np.dot(w.T, X))).mean(axis=1) - _g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new


def jade(X):

    m, n = X.shape
    W = np.eye(n)

    for i in range(m - 1):
        for j in range(i + 1, m):
            
            # compute the Givens rotation
            x1, x2 = X[i, :], X[j, :]
            G = np.eye(n)
            d = np.sqrt(x1.dot(x1) + x2.dot(x2))
            c = x1.dot(x2) / d
            s = np.sqrt(1 - c**2)
            G[[i, j], [i, j]] = c
            G[i, j] = s
            G[j, i] = -s

            # update the estimated eigenmatrix
            X = G.dot(X)
            W = W.dot(G.T)

    return W
