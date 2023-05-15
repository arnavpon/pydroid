"""
Module applying Wavelet algorithm to given signals.

March 6, 2023
"""

import numpy as np
import pywt


def apply_wavelet(signal, wave = 'db2', level = 1):
    """
    Given a signal, apply wavelet transform to it and return a
    resulting signal. Wavelet transform removes noise from the signal.
    Resultant filtered signal is upsampled so that the signal that is
    returned has the same length as the original signal.
    """

    # apply the wavelet transform, repeatedly according the the number of levels given
    filtered_signal = _wavelet_denoise(signal, wave, level)

    # interpolate the filtered signal to match the length of the original signal
    x_old = np.linspace(0, 1, len(filtered_signal))
    x_new = np.linspace(0, 1, len(signal))
    return np.interp(x_new, x_old, filtered_signal)


def _wavelet_denoise(signal, wavelet, level):
    
    # track signal at the end of each level
    level_results = []
    sig = signal.copy()
    for _ in range(level):
        sig, _ = pywt.dwt(sig, wavelet)
        level_results.append(sig)

    return level_results[-1]

def cwt_morlet_filter(signal, sample_freq = 30, f_min = 0.67, f_max = 4.0, num_steps = 200):

    sampling_period = 1.0 / sample_freq

    # Define the wavelet
    wavelet = 'morl'

    # Compute the corresponding scales
    scale_max = pywt.scale2frequency(wavelet, f_min) * sampling_period
    scale_min = pywt.scale2frequency(wavelet, f_max) * sampling_period

    # Create a range of scales
    scales = np.linespace(scale_min, scale_max, num_steps)

    # Compute the CWT
    cwt_coeffs, _ = pywt.cwt(signal, scales, wavelet, sampling_period)

    # Filter the signal
    filtered_signal = np.sum(cwt_coeffs, axis=0)

    return filtered_signal
