from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from extract_features import get_features, get_features_from_signals_peaks
import numpy as np
from wavelet_util import *
    
class Unsupervised:

    def __init__(self, X, model = 'KM', nclust = 5):
        self.X = X

        if model != 'KM':
            raise ValueError('Model not supported')
        
        self.model_name = model
        self.model = KMeans(n_clusters=nclust, random_state=0)
    
    def train(self):
        self.model.fit(self.X)
    
    def predict(self, X):
        return self.model.predict(X)

    def save(self):
        with open(f'{self.model_name}.sav', 'wb') as f:
            pickle.dump(uns, f)
        f.close()

class MLP:
    
    def __init__(self, X, y, hidden_layer_sizes = (100, 50)):
        self.X = X
        self.y = y
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
    
    def train(self):
        self.model.fit(self.X, self.y)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        with open(f'mlp.sav', 'wb') as f:
            pickle.dump(uns, f)
        f.close()


if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt

    s1 = 1000
    s2 = 4000
    
    signal, _ = good_get_signal_peaks('001', '001')
    signal = signal[s1: s2]
    
    den = wavelet_denoise(signal, wavelet='db10', level = 2)[0]
    cA_peaks = get_wavelet_peaks(den)

    def _normalize_signal(signal, target_amplitude):
        max_amplitude = np.max(np.abs(signal))
        scale_factor = target_amplitude / max_amplitude
        normalized_signal = signal * scale_factor
        return normalized_signal

    feats = get_features_from_signals_peaks(den, cA_peaks)
        
    for subj in range(2, 8):
        
        signal, _ = good_get_signal_peaks(f'00{subj}', '001')
        signal = signal[s1: s2]

        den = wavelet_denoise(signal, wavelet='db10', level = 2)[0]
        cA_peaks = get_wavelet_peaks(den)
        
        fo = get_features_from_signals_peaks(den, cA_peaks)
        feats = np.concatenate((feats, fo), axis = 0)

    feats = feats.astype(np.float64)
    uns = Unsupervised(feats)
    uns.train()
    labels = uns.predict(feats) != 2

    X_train, X_test, y_train, y_test = train_test_split(
        feats, labels, test_size = 0.33
    )

    mlp = MLP(X_train, y_train)
    mlp.train()
    res = mlp.predict(X_test)
    
    res = [1 if x > 0.5 else 0 for x in res]
    print('Precision: ', precision_score(y_test, res))
    print('Recall: ', recall_score(y_test, res))
    print('F1: ', f1_score(y_test, res))

    mlp.save()
    

    
    
