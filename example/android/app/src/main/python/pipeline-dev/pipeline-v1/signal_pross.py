"""
Signal processing module for cleaning up rPPG signals.

March 6, 2023
"""

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
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
    
def normalize_amplitude_to_1(signal: np.ndarray):
    """
    Normalize given signal to given amplitude.
    """

    sigmax = abs(max(signal))
    sigmin = abs(min(signal))
    
    return np.array([
        v / sigmax if v > 0 else v / sigmin
        for v in signal
    ])


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

def min_max_scale(signal: np.ndarray, min_val: float = 0, max_val: float = 1):
    """
    Scale the given signal to the range [min_val, max_val].
    """

    min_signal = np.nanmin(signal)
    max_signal = np.nanmax(signal)
    return (signal - min_signal) * (max_val - min_val) / (max_signal - min_signal) + min_val

def normalize_signal(signal: np.ndarray):
    """
    Normalize the given signal using mean and std.
    """

    mn = np.nanmean(signal)
    std = np.nanstd(signal)
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

def get_ibis(peaks, fr = 30, with_valleys = False):
    ibis = []
    for i in range(1, len(peaks)):

        if with_valleys:
            peak_diff = peaks[i][0] - peaks[i - 1][0]
        else: 
            peak_diff = peaks[i] - peaks[i - 1]
        
        ibi = peak_diff / fr

        if with_valleys and peaks[i][1] != peaks[i - 1][1]:
            ibi *= 2
        
        ibis.append(ibi)

    return ibis

def get_hr(ibis):
    return 60 / np.mean(ibis)


def get_hrv(ibis):
    differences = np.diff(ibis)
    squared_diffs = np.square(differences)
    mean_squared_diffs = np.mean(squared_diffs)
    rmssd = np.sqrt(mean_squared_diffs)
    return rmssd


def get_hr_from_fourier(signal, fr, min_freq = 0.7, max_freq = 3):
    
    signal_size = len(signal)
    signal = signal.flatten()
    
    fft_data = np.fft.rfft(signal) # FFT
    fft_data = np.abs(fft_data)

    freq = np.fft.rfftfreq(signal_size, 1. / fr) # Frequency data

    inds = np.where((freq < min_freq) | (freq > max_freq))[0]
    fft_data[inds] = 0
    bps_freq=60.0*freq
    max_index = np.argmax(fft_data)
    fft_data[max_index] = fft_data[max_index]**2
    HR =  bps_freq[max_index]
    return HR
