"""
Module applying Wavelet algorithm to given signals.

March 6, 2023
"""

import pywt


def apply_wavelet(signal, wave = 'db2', level = 1):
    """
    Given a signal, apply wavelet transform to it and return a
    resulting signal.
    """

    # apply the wavelet transform, repeatedly according the the number of levels given
    return _wavelet_denoise(signal, wave, level)


def _wavelet_denoise(signal, wavelet, level):
    
    # track signal at the end of each level
    vs = []
    sig = signal.copy()
    for _ in range(level):
        sig, cD = pywt.dwt(sig, wavelet)
        vs.append(sig)

    return vs[-1]
