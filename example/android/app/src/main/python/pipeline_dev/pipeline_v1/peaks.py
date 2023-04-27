"""
Peak detection module for rPPG.

March 6, 2023
"""

import numpy as np
from scipy.signal import find_peaks, peak_prominences


def get_peaks_for_hr(signal: np.ndarray, fr: int, max_freq: float, peak_height: float,
    prominence: float = 0, with_valleys: bool = False,
    with_min_dist: bool = True):
    """
    Find peaks in the given signal. Filter for minumum distance from each other and a 
    minimum prominance threshold.

    The use of valleys didn't end up making it into the thesis. The idea was that
    we could potentially double the amount of good signal we had by multiplying the signal by -1 and
    using the new "peaks" as further peaks, but this ultimately didn't improve performance.
    """
    
    # skip any None values at the beginning of the signal
    first_index = None
    for i in range(len(signal)):
        if signal[i] is not None:
            first_index = i
            break
    signal = signal[first_index: ]

    def _filter_peaks(sig):
        min_dist = fr // max_freq if with_min_dist else 1
        peaks, _ = find_peaks(sig, height = peak_height, prominence = prominence, distance = min_dist)
        prominences = {p + first_index: prom for p, prom in zip(peaks, peak_prominences(sig, peaks)[0])}
        return [p + first_index for p in peaks], prominences

    peaks, peak_proms = _filter_peaks(signal)
    
    if with_valleys:
        valleys, valley_prominences = _filter_peaks(-signal)
        peaks = sorted(list(set(peaks + valleys)))
        peak_proms = {**peak_proms, **valley_prominences}
        return peaks, peak_proms

    return np.array(peaks), peak_proms
