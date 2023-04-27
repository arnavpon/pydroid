"""
Module applying Wavelet algorithm to given signals.

March 6, 2023
"""

import numpy as np
import pywt
from scipy.signal import butter, filtfilt


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
