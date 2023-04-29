"""
Module implementing a suite of new features to extract from the forehead ROI.

April 13, 2023
"""


import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew


def gradient_feature(roi):

    features = []
    for color in range(3):
        
        x_grad = np.gradient(roi[:, :, color])[0]
        y_grad = np.gradient(roi[:, :, color])[1]
        
        grad_mag = np.sqrt(np.power(x_grad, 2) + np.power(y_grad, 2))
        max_grad_mag = np.max(grad_mag)
        features.append(max_grad_mag)
    
    return np.array(features)


def skewness_kurtosis(roi):

    features = []
    for color in range(3):
        
        skewness = skew(roi[:, :,color])
        kurtosis = kurtosis(roi[:, :, color])
        features.append((skewness, kurtosis))
    
    return np.array(features)


def entropy_feature_single_frame(roi, num_bins = 16):

    features = []
    for color in range(3):
        
        data = roi[:, :, color].reshape(-1)
        histogram, _ = np.histogram(data, bins = num_bins, density = True)
        histogram = histogram + 1e-8
        
        entropy = -1 * np.sum(histogram * np.log2(histogram))
        features.append(entropy)
    
    return np.array(features)


def spectral_features(roi):

    features = []
    for color in range(3):
        
        f, psd = welch(roi[:, :, color].flatten())
        peakfreq = f[np.argmax(psd)]
        features.append(peakfreq)
        features.append(np.sum(psd))
    
    return features
