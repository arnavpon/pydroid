import numpy as np
from jadeR import jadeR
np.random.seed(0)


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


def jade_v2(X):
    N, T = X.shape
    C = np.cov(X)
    eigen_values, eigen_vectors = np.linalg.eig(C)
    order = np.argsort(eigen_values)[::-1]
    W = eigen_vectors[:, order[:3]]
    return W


def jade_v3(X):

    m, n = X.shape
    W = np.eye(m)

    for i in range(n - 1):
        for j in range(i + 1, n):
            print('X begins as')
            print(X)
            
            # compute the Givens rotation
            x1, x2 = X[:, i], X[:, j]
            print('x1 is', x1)
            G = np.eye(m)
            d = np.sqrt(x1.dot(x1) + x2.dot(x2))
            c = x1.dot(x2) / d
            print('c is', c)
            
            c_res = 1 - c**2
            s = np.sqrt(c_res) if c_res > 0 else 0
            print('s is ', s)

            if x1[0] < 0:
                s = -s
            
            G[i, i], G[j, j] = c, c
            G[i, j] = s
            G[j, i] = -s
            
            print('G is')
            print(G)

            # update the estimated eigenmatrix and mixing matrix
            X = G.dot(X)
            print('X updated to')
            print(X)
            W = W.dot(G.T)


            print()

    return W


def jade_v4(X):
    return jadeR(X.T, m = X.shape[1]).T