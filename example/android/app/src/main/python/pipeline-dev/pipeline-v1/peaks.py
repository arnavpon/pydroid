"""
Peak detection module for rPPG.

March 6, 2023
"""

import numpy as np
from scipy.signal import find_peaks

    
def get_peaks(signal: np.ndarray, fr: int, max_freq: float, peak_height: float,
    slice_filter_thresh: int, perc1: float, perc2: float,
    with_min_dist: bool = True, with_additional_filtering: bool = True):
    """
    The current standard method for peak detection. Includes the option to use the _filter_peaks
    method for even more aggressive filtering.
    """

    if with_min_dist:
        min_dist = fr // max_freq
    else:
        min_dist = 1
    
    peaks, _ = find_peaks(signal, height = peak_height, distance = min_dist)

    if with_additional_filtering:
        peaks = _filter_peaks(signal, peaks, fr, slice_filter_thresh, perc1, perc2)
    
    return peaks


def _filter_peaks(signal: np.ndarray, peaks: np.ndarray, fr: int,
    slice_filter_thresh: int, perc1: float, perc2: float):
    """
    Filter peaks with the intent of trying to peaks that are "definitely" noise.
    """

    # first, remove peaks that aren't sufficienrtly above the perc2 percentile. I intentionally
    # am comparing the peaks to the entire array, and not just the set of peaks, because I'm not
    # making the assumption that a certain percentage of peaks will inherently be noisy. However,
    # I am making the assumption that if the peak is insufficiently clear of a certain baseline of
    # the entire signal, then it must be noise.
    peaks = np.array([p for p in peaks if signal[p] >= np.percentile(signal, perc2)])   
    
    # Peak Walk: remove peaks that are close together, that aren't sufficiently large relative to the
    # entire signal. This percentile (perc1) is more stringent than the one used in the first step.
    for i in range(0, len(signal) - fr, fr):
        
        j = i + fr
        slce = peaks[(peaks >= i) & (peaks < j)]
        
        if len(slce) > slice_filter_thresh:
            to_remove = [i for i in range(len(slce)) if signal[slce[i]] < np.percentile(signal[slce], perc1)]
            peaks = peaks[~np.isin(peaks, [slce[i] for i in to_remove])]
            removed += len(to_remove)
    
    return peaks
