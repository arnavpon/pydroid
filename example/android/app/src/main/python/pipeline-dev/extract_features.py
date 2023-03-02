import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import simps
from sklearn.mixture import GaussianMixture
from scipy.stats import skew, kurtosis
from tqdm import tqdm

from wavelet_util import *


def get_features(subj, trial):
    signal, peaks = good_get_signal_peaks(subj, trial)
    return get_features_from_signals_peaks(signal, peaks)

def get_features_from_signals_peaks(signal, peaks): 

    def _normalize_signal(signal, target_amplitude):
        max_amplitude = np.max(np.abs(signal))
        scale_factor = target_amplitude / max_amplitude
        normalized_signal = signal * scale_factor
        return normalized_signal
    
    norm = _normalize_signal(signal, 1)
    features = extract_peak_features(norm, peaks)
    features = normalize_data(features)
    return features

def extract_peak_features(sig, peaks):
    features = []
    for idx in tqdm(range(len(peaks))):
        pidx = peaks[idx]
        
        # peak = signal_array[index-10:index+10]  # extract a window of 20 samples around the peak
        amplitude = np.abs(sig[pidx])
        
        def _get_mount(signal, peak_idx):
            
            left_idx = None
            for i in range(peak_idx - 1, -1, -1):
                if signal[i] > signal[i + 1]:
                    left_idx = i + 1
                    break
            if left_idx is None:
                left_idx = 0
            
            right_idx = None
            for i in range(peak_idx + 1, len(signal)):
                if signal[i] > signal[i - 1]:
                    right_idx = i - 1
                    break
            if right_idx is None:
                right_idx = len(signal) - 1
            
            return signal[left_idx: right_idx + 1]


        mount = _get_mount(sig, pidx)
        
        gmm = GaussianMixture(n_components=1, covariance_type='diag', tol=1e-6, max_iter=10000)
        gmm.fit(mount.reshape(-1, 1))
        mean = gmm.means_[0][0]
        std = np.sqrt(gmm.covariances_[0][0])
        fwhm = 2 * np.sqrt(2 * np.log(2)) * std  # FWHM of the peak
        
        area = simps(mount)  # area under the curve of the peak using Simpson's rule
        skewness = skew(mount)  # skewness of the peak
        kurt = kurtosis(mount)  # kurtosis of the peak
        slope = np.max(np.abs(np.diff(mount)))  # slope of the peak
        zero_crossing_rate = len(np.where(np.diff(np.sign(mount)))[0])  # zero-crossing rate of the peak
        
        feature_vector = [amplitude, fwhm, area, skewness, kurt, slope, zero_crossing_rate]
        features.append(feature_vector)
    
    return np.array(features)

def normalize_data(data):
    """
    Normalizes the data using z-score normalization.

    Args:
    data: a 2D Numpy array of shape (n_samples, n_features)

    Returns:
    normalized_data: a 2D Numpy array of shape (n_samples, n_features), normalized using z-score normalization
    """
    # Compute the mean and standard deviation for each feature
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    # Normalize the data using z-score normalization
    normalized_data = (data - means) / stds

    return normalized_data


if __name__ == '__main__':

    feat = get_features('001', '001')
    print(feat.shape)