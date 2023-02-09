"""
Module handling the calculation of HR and HRV using color channel vectors
derived from a face video.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline, interp1d, InterpolatedUnivariateSpline as Spline
from scipy.signal import welch, stft, istft, windows, butter, filtfilt, hamming, find_peaks, lfilter
from scipy.sparse import diags

from ica import jade_v4
from tracking import DEFAULT_CSV_NAME


def load_channels(path = DEFAULT_CSV_NAME):
    """
    Load ROI channel vectors from local csv file.
    """
    
    channels = pd.read_csv(path)
    if (
        'r' not in channels.columns
        or 'g' not in channels.columns
        or 'b' not in channels.columns
    ):
        raise Exception('Channels data must contain r, g, and b data.')

    return {
        'r': channels.r.to_numpy(),
        'g': channels.g.to_numpy(),
        'b': channels.b.to_numpy()
    }


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

# ======= ICA =======

def get_bvp_w_ica(X, ica_method = jade_v4):

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


def pipeline(path = DEFAULT_CSV_NAME, 
            detrend_method = detrend_w_poly,
            ica_method = jade_v4,
            moving_average_window = 15):

    # Step 1: load the spatially averaged color channels from the video
    # NOTE: Idea 1: apply signal processing on the pixel level, instead of
    # spatially averaging first
    channels = load_channels(path = path)

    # Step 2: Detrend each channel individually
    channels = {
        'r': detrend_method(channels['r']),
        'g': detrend_method(channels['g']),
        'b': detrend_method(channels['b'])
    }

    # Step 3: Normalize each channel
    channels = {
        'r': normalize_detrended(channels['r']),
        'g': normalize_detrended(channels['g']),
        'b': normalize_detrended(channels['b'])
    }

    plt.title('Normalized Detrended Channels')
    for k in channels:
        plt.plot(channels[k])
    plt.show()

    # Step 4: Apply ICA and get component with highest spectrum peak
    X = np.stack([channels['r'], channels['g'], channels['b']], axis = 1)
    #bvp_comp = get_bvp_w_ica(X, ica_method = ica_method)
    bvp_comp = channels['g']

    plt.title('ICA Selected Component')
    plt.plot(bvp_comp)
    plt.show()

    # Step 5: Apply 5-point moving average filter to the peak comp
    fcomp = n_moving_avg(bvp_comp, window = moving_average_window)
    plt.title('5-point Moving Average Filtered Component')
    plt.plot(fcomp)
    plt.show()

    peaks = get_peaks(fcomp)
    print('peaks len', len(peaks))
    a = [fcomp[i] for i in peaks]
    plt.plot(fcomp)
    plt.scatter(peaks, [fcomp[i] for i in peaks], marker = 'x', color = 'red')
    plt.show()

    ibis = get_ibis(peaks)
    plt.title('IBIs')
    plt.plot(ibis)
    plt.show()

    mibis = np.mean(ibis)
    print('Mean IBI:', mibis)
    print('HR:', get_hr(ibis))


if __name__ == '__main__':
    pipeline(path = '/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/subj_010-channel_data.csv')
