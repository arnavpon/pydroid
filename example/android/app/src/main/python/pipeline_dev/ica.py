from jadeR import jadeR

def jade(X):
    return jadeR(X.T, m = X.shape[1]).T
