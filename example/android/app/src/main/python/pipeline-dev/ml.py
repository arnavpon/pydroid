import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def extract_features(signal, window_size):
    n_windows = signal.shape[0] // window_size
    
    features = []
    
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        
        window = signal[start:end]
        
        # Calculate features for the window
        window_mean = np.mean(window)
        window_std = np.std(window)
        window_skewness = stats.skew(window)
        window_kurtosis = stats.kurtosis(window)
        window_max = np.max(window)
        window_min = np.min(window)
        window_slope = np.polyfit(np.arange(window_size), window, 1)[0]
        
        features.append([window_mean, window_std, window_skewness, window_kurtosis, window_max, window_min, window_slope])
    
    return np.array(features)



df = pd.read_csv('validation_data/IEEE_data/subject_001/trial_001/empatica_e4/IBI.csv')
l = df[' IBI'].tolist()
plt.plot(l)
plt.show()
