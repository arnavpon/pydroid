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


# def bandpass(arr, hamming_window, freq_range):

#     # Generate a Hamming window of the specified size
#     window = windows.hann(hamming_window)

#     # use short-time Fourier transform (STFT) on the arr
#     # with hamming window of given size to get the frequency, time, and STFT results
#     f, t, Zxx = stft(arr, window = window, nperseg = hamming_window)
#     # print('sam', Zxx.shape)
#     # plt.plot(Zxx)
#     # plt.show()

#     # copy the STFT results
#     Zxx_filtered = np.copy(Zxx)

#     # iterate over each time step in the STFT results
#     for i in range(Zxx.shape[1]):
        
#         # iterate over each freq in the STFT results
#         for j in range(Zxx.shape[0]):
            
#             # check if the freq is outside the given freq range, 
#             # and set corresponding Zxx copy index to 0 if so
#             abs_fj = abs(f[j])
#             print(freq_range, abs_fj)
#             if abs_fj < freq_range[0] or abs_fj > freq_range[1]:
#                 Zxx_filtered[j][i] = 0
#             else: print('escaped', abs_fj)

#     # use inverse STFT to obtain the filtered signal
#     _, arr_filtered = istft(Zxx_filtered, window = window, nperseg = hamming_window)
#     return arr_filtered


# def bandpass_filter(arr, low, high, fs=30, order=128):
#     nyquist = fs / 2
#     low = low / nyquist
#     high = high / nyquist

#     b, a = butter(order, [low, high], btype='band')
#     window = hamming(order + 1)
#     filtered = filtfilt(b, a, arr * window)
#     return filtered

# def bpf(data, low, high, fs=30, order=3):
#     nyquist = fs / 2
#     low = low / nyquist
#     high = high / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     window = hamming(128)
#     filtered = np.zeros(len(data))
#     for i in range(0, len(data) - 128, 64):
#         segment = data[i:i + 128]
#         filtered[i:i + 128] = filtfilt(b, a, segment * window)
#     return filtered

# ======= Peak Detection =======

def get_peaks(arr, fr = 30, thresh = 0.5, min_accepted_hr = 20, max_accepted_hr = 120):
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

def csi(signal):
    x = np.arange(len(signal))
    y = signal
    sample_frequency = 256
    x_new = np.linspace(0, x[-1], int(x[-1] * sample_frequency))
    spline = CubicSpline(x, y)
    y_new = spline(x_new)
    return y_new
    

def cubic_spline_interpolation(signal, sample_frequency=256):
    time = np.arange(0, len(signal)/sample_frequency, 1/sample_frequency)
    spline = CubicSpline(time, signal)
    new_time = np.arange(0, len(signal) / sample_frequency, 1 / sample_frequency)
    new_time = np.clip(new_time, time.min(), time.max())
    new_signal = spline(new_time)
    return new_signal

def get_ibis_w_bvp_peak_detection(arr):
    
    # init an empty list to store the peak locations
    peak_locs = []

    # iterate over the interpolated signal to detect BVP peaks
    len_arr = len(arr)
    for i in range(1, len_arr - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peak_locs.append(i)

    # apply NC-VT filtering to the IBIs
    return nc_vt_filter(peak_locs)

def nc_vt_filter(peak_locs, tolerance = 0.3):
    
    # init an empty list to store the IBIs
    ibis = []

    # apply NC-VT filtering to the IBIs
    for i in range(1, len(peak_locs) - 1):
        
        curr_ibi = peak_locs[i + 1] - peak_locs[i]
        prev_ibi = peak_locs[i] - peak_locs[i - 1]
        
        if (curr_ibi / prev_ibi) >= 1 - tolerance and (curr_ibi / prev_ibi) <= 1 + tolerance:
            ibis.append(curr_ibi)

    return ibis


def check_power_spectrum(signal, sample_rate):
    # Compute the FFT of the signal
    fft = np.fft.fft(signal)
    
    # Get the frequencies corresponding to the FFT coefficients
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    # Compute the power spectrum by taking the absolute value squared of the FFT
    power_spectrum = np.abs(fft)**2
    
    # Plot the power spectrum
    plt.plot(freqs, power_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.show()

def get_interbeat_intervals(signal):
    peaks, _ = find_peaks(signal)
    ibis = np.diff(peaks)
    return ibis

def filter_interbeat_intervals(interbeat_intervals):
    mean_ibi = sum(interbeat_intervals) / len(interbeat_intervals)
    tolerance = mean_ibi * 0.3
    filtered_intervals = []
    for i in range(len(interbeat_intervals)):
        if (mean_ibi - tolerance) <= interbeat_intervals[i] <= (mean_ibi + tolerance):
            filtered_intervals.append(interbeat_intervals[i])
    return filtered_intervals


def get_hr(ibis):
    return 60 / np.mean(ibis)


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
    bvp_comp = get_bvp_w_ica(X, ica_method = ica_method)

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

    # Step 6: Apply bandpass filter to the component
    # fcomp = bandpass(fcomp, 0.7, 4)
    # # fcomp = bpf(fcomp, 0.7, 4)
    # plt.title('Bandpass Filtered Component')
    # plt.plot(fcomp)
    # plt.show()

    # # # Step 7: Interpolate signal w/ cubic spline function
    # interp = cubic_spline_interpolation(fcomp)
    # ibis = get_interbeat_intervals(interp)

    # ibis = filter_interbeat_intervals(ibis)

    # # each ibi value represents a number of frames
    # fr = 30
    # ibis = [ibi / fr for ibi in ibis]
    # print(np.mean(ibis))

    # plt.plot(ibis)
    # plt.show()
    # # # # Step 8: Compute IBIs
    # # ibis = get_ibis_w_bvp_peak_detection(icomp)
    # # # plt.plot(ibis)
    # # # plt.show()

    # # # # Step 9: Return HR
    # return get_hr(ibis)


if __name__ == '__main__':

    print('HR is', pipeline())
