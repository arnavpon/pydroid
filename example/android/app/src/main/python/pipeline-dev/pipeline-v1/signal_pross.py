"""
Signal processing module for cleaning up rPPG signals.

March 6, 2023
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple


def bandpass(signal: np.ndarray, fr: int, freq: Tuple[float, float], order: int):
    """
    Apply bandpass filter to the given signal.

    fr - frame rate
    freq - tuple of low and high frequencies for the bandpass filter
    order - order of the bandpass filter
    """

    # nyquist frequency stays hardcoded at half the frame rate
    nyquist_freq = 0.5 * fr
    
    # low and high values for butter created using nyquist
    low = freq[0] / nyquist_freq
    high = freq[1] / nyquist_freq
    
    # apply the filter
    b, a = butter(order, [low, high], btype = 'band')
    filtered = filtfilt(b, a, signal)
    return filtered
    
def normalize_to_amplitude(signal: np.ndarray, target_amplitude: float):
    """
    Normalize given signal to given amplitude.
    """
    
    sig_max = np.max(np.abs(signal))
    scale_factor = target_amplitude / sig_max
    return signal * scale_factor


def n_moving_avg(signal: np.ndarray, window: int = 5):
    """
    Simple moving window smoothing for a given signal.
    """

    result = []
    for i in range(len(signal) - (window - 1)):
        result.append(
            float(sum(signal[i: i + window])) / window
        )
    
    return np.array(result)

def normalize_signal(signal: np.ndarray):
    """
    Normalize the given signal using mean and std.
    """

    mn = np.mean(signal)
    std = np.std(signal)
    return (signal - mn) / std

def detrend_w_poly(signal: np.ndarray, degree: int = 3):
    """
    Detrend signal using nth degree polynomial.
    """

    siglen = len(signal)
    x = np.arange(siglen)
    poly = np.polyfit(x, signal, degree)
    curve = np.poly1d(poly)(x)
    return signal - curve

def get_ibis(peaks, fr = 30):
    ibis = []
    for i in range(1, len(peaks)):
        ibis.append((peaks[i] - peaks[i - 1]) / fr)
    return ibis

def get_hr(ibis):
    return 60 / np.mean(ibis)
