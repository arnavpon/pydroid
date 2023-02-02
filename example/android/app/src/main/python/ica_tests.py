import numpy as np
from ica import jade

def test_jade():
    np.random.seed(0)
    eps = 1e-8

    def test_orthogonal_matrices():
        X = np.random.rand(100, 3)
        X, _ = np.linalg.qr(X)
        result = jade(X)
        assert np.allclose(result, np.eye(3), atol=eps)
        
    def test_correlated_matrices():
        X = np.random.rand(100, 3)
        X[:, 0] += 2 * X[:, 1]
        X[:, 2] += X[:, 1]
        result = jade(X)
        assert np.allclose(np.diag(result), np.ones(3), atol=eps)
        assert np.abs(result[0, 1]) < eps
        assert np.abs(result[0, 2]) < eps
        assert np.abs(result[1, 2]) < eps
        
    def test_large_correlation():
        X = np.random.rand(100, 3)
        X[:, 0] += 100 * X[:, 1]
        X[:, 2] += 1000 * X[:, 1]
        result = jade(X)
        assert np.allclose(np.diag(result), np.ones(3), atol=eps)
        assert np.abs(result[0, 1]) < eps
        assert np.abs(result[0, 2]) < eps
        assert np.abs(result[1, 2]) < eps
        
    def test_identical_matrices():
        X = np.ones((100, 3))
        result = jade(X)
        assert np.allclose(np.diag(result), np.ones(3), atol=eps)
        assert np.abs(result[0, 1]) < eps
        assert np.abs(result[0, 2]) < eps
        assert np.abs(result[1, 2]) < eps


    # "Main"  
    test_orthogonal_matrices()
    test_correlated_matrices()
    test_large_correlation()
    test_identical_matrices()
    

if __name__ == '__main__': 
    test_jade()
