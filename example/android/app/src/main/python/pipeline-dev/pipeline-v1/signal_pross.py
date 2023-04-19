"""
Signal processing module for cleaning up rPPG signals.

March 6, 2023
"""

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
from sklearn.decomposition import FastICA
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
    denom = (max_signal - min_signal) + min_val
    if denom == 0:
        return signal
    return (signal - min_signal) * (max_val - min_val) / denom

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

    return np.array(ibis)

def get_hr(ibis):
    return 60 / np.mean(ibis)


def get_hrv(ibis):
    """
    HRV via the RMSSD method.
    """

    mean_sq_diffs = np.mean(np.square(np.diff(ibis)))
    return np.sqrt(mean_sq_diffs)


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


def perform_ica(raw_traces, n_components=3):
    """
    Perform Independent Component Analysis (ICA) using the FastICA algorithm on the given normalized raw traces.

    Parameters:
    raw_traces (numpy.ndarray): A 2D array of normalized raw traces with shape (n_samples, n_channels).
    n_components (int, optional): Number of independent components to extract. Default is 3.

    Returns:
    numpy.ndarray: The selected independent component with the highest peak in its power spectrum.
    """
    # Perform ICA using the FastICA algorithm
    ica = FastICA(n_components=n_components)
    source_signals = ica.fit_transform(raw_traces)

    # Calculate power spectra of the independent components
    power_spectra = np.abs(np.fft.fft(source_signals, axis=0))**2

    # Find the component with the highest peak in its power spectrum
    max_peak_component = None
    max_peak_height = -np.inf
    for i in range(n_components):
        peaks, _ = find_peaks(power_spectra[:, i])
        peak_height = np.max(power_spectra[peaks, i])
        if peak_height > max_peak_height:
            max_peak_height = peak_height
            max_peak_component = i

    return source_signals[:, max_peak_component]
