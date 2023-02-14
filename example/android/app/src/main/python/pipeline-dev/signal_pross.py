"""
Module for signal processing utility methods to be used in the HR pipeline.
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d, InterpolatedUnivariateSpline as Spline
from scipy.signal import welch, stft, istft, windows, butter, filtfilt, hamming, find_peaks, lfilter
from scipy.sparse import diags

from ica import jade


# ======= Detrending Methods =======

def detrend(channel, smoothing_param = 10):
    """
    Given a color channel vector, apply detrending as described in 
    "An Advanced Detrending Method With Application to HRV Analysis"
    """

    # make sure the channel is a np array
    if not isinstance(channel, np.ndarray):
        channel = np.array(channel)
    
    # get length of the vector
    T = len(channel)

    I = np.eye(T)
    D2 = diags(
        [1, -2, 1],
        [0, 1, 2],
        shape = (T - 2, T)
    ).toarray()

    z_stat = np.dot(
        np.eye(T) - np.linalg.inv(I + (smoothing_param ** 2) * np.dot(D2.T, D2)),
        channel
    )

    return z_stat

def detrend_by_differencing(channel):
    """
    Step 2: Detrend by differencing. This is the simplest detrending method.
    """

    # make sure the channel is a np array
    if not isinstance(channel, np.ndarray):
        channel = np.array(channel)

    clen = len(channel)
    res = np.zeros(clen)
    res[0] = channel[0]
    for i in range(1, clen):
        res[i] = channel[i] - channel[i - 1]

    return res


def detrend_w_poly(channel, degree = 3):
    """
    Detrend using nth degree polynomial.

    NOTE: This is the detrend method to use for now.
    """

    # make sure the channel is a np array
    if not isinstance(channel, np.ndarray):
        channel = np.array(channel)

    clen = len(channel)
    x = np.arange(clen)
    poly = np.polyfit(x, channel , degree)
    curve = np.poly1d(poly)(x)
    return channel - curve

# ======= End Detrending Methods =======

# ======= Normalization =======

def normalize_detrended(detrended_channel):
    """
    Input is detrended channel. This method implements
    formula 3 in "Advancements in Noncontact, Multiparameter Physiological
    Measurements Using a Webcam".
    """

    # make sure the channel is a np array
    if not isinstance(detrended_channel, np.ndarray):
        detrended_channel = np.array(detrended_channel)

    # get mean and standard dev of the channel
    mn = np.mean(detrended_channel)
    std = np.std(detrended_channel)

    return (detrended_channel - mn) / std

# ======= End Normalization =======

# ======= ICA and signal selection =======

def get_bvp_w_ica(X, ica_method = jade):

    # decompose normalized raw traces
    mixing_matrix = ica_method(X)

    components = np.copy(X).dot(mixing_matrix)

    power_spectra = []
    for i in range(components.shape[1]):
        _, Pxx = welch(components[:, i], fs = 1000)
        power_spectra.append(Pxx)

    maxes = [np.max(ps) for ps in power_spectra]
    bvp_index = maxes.index(max(maxes))
    return components[:, bvp_index]

# ======= End ICA =======

# ======= Filtering =======

def n_moving_avg(arr, window = 5):
    """
    Step 5: Simple N-point moving average method. Will default to window
    of 5, as that is what's used in the paper.
    """

    result = []
    for i in range(len(arr) - 4):
        result.append(float(sum(arr[i: i + window])) / window)
    
    return result

def bandpass(data, lowcut, highcut, fs = 30, order = 5):
    b, a = _bandpass_helper(lowcut, highcut, fs, order = order)
    y = lfilter(b, a, data)
    return y

def _bandpass_helper(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs = fs, btype='band')


# ======= End Filtering =======

# ======= Peak Detection =======

def get_peaks_simple(arr):
    """
    Step 6: Peak detection. This is the simplest peak detection method.
    """

    peak_idxs = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            # if abs(arr[i]) > thresh:
            peak_idxs.append(i)
    
    return peak_idxs

def get_peaks(arr, fr = 30, thresh = 0.25, min_accepted_hr = 20, max_accepted_hr = 120):
    """
    Step 6: Peak detection. Add a min distance between peaks based
    on reasonable assumptions about the possible range of heart rates.
    I'm assuming the min reasonable heart rate is 20 and the max is 200.
    The amount of frames per beat is fr*60/HR, so for a frame rate of 
    30, max heart rate of 200, the min distance between peaks is 30*60/200=9.
    """

    min_dist = fr * 60 / max_accepted_hr

    peak_idxs = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            if (
                (abs(arr[i]) > thresh)
                and (len(peak_idxs) == 0 or (len(peak_idxs) > 0 and i - peak_idxs[-1] >= min_dist))
            ):
                peak_idxs.append(i)

    return peak_idxs

def ncvt(signal, tolerance = 0.3, step = 50):

    uu = tolerance
    um = tolerance

    L = 0
    M = 1e-10  # small nonzero value

    result = [signal[0]]
    for i in range(1, len(signal) - 1):

        if (
            abs(signal[i] - signal[L]) / signal[L] < uu
            or abs(signal[i] - signal[i + 1]) / signal[L] < uu
            or abs(signal[i] - M) / M < um
        ):
            
            result.append(signal[i])
            L = i
            
            if len(result) % step == 0:
                M = np.mean(result[-step: ])
                uu = tolerance / M
    
    return result

# ======= End Peak Detection =======

# ===== Interbeat intervals (IBI) and HR estimation =====

def get_ibis(peaks, fr = 30):
    """
    Step 7: Get the interbeat intervals (IBIs) from the peak indices.
    """

    ibis = []
    for i in range(1, len(peaks)):
        ibis.append((peaks[i] - peaks[i - 1]) / fr)
    return ibis

def get_hr(ibis):
    return 60 / np.mean(ibis)

# ===== End Interbeat intervals (IBI) and HR estimation =====
