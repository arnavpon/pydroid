import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences
import sys
from typing import Tuple

from truth import IeeeGroundTruth

plt.rcParams['figure.figsize'] = [18, 6]

peak_height = 0.25
CHROM_SETTINGS = {
    'fr': 30,  # frame rate
    'freq': (0.67, 3.0),  # bandpass frequency range
    'bandpass_order': 4,  # bandpass filter order
    'moving_avg_window': 6,  # moving average window size for smoothing
    'peak_height': peak_height,  # min peak height for peak detection
    'slice_filter_thresh': 2,  # min number of peaks allowed in a slice of the signal
    'stringent_perc': 85,  # more stringent percentile for peak filtering
    'non_stringent_perc': 75,  # less stringent percentile for peak filtering,
    'prominence': 0.15
}


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

def detrend_w_poly(signal: np.ndarray, degree: int = 3):
    """
    Detrend signal using nth degree polynomial.
    """

    siglen = len(signal)
    x = np.arange(siglen)
    poly = np.polyfit(x, signal, degree)
    curve = np.poly1d(poly)(x)
    return signal - curve

def normalize_signal(signal: np.ndarray):
    """
    Normalize the given signal using mean and std.
    """

    mn = np.mean(signal)
    std = np.std(signal)
    return (signal - mn) / std

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

def normalize_amplitude_to_1(signal: np.ndarray):
    """
    Normalize amplitude of given signal to 1.
    """
    
    # skip any None values at the beggining of the signal
    first_index = None
    for i in range(len(signal)):
        if signal[i] is not None:
            first_index = i
            break
    
    first_part = list(signal[0: first_index])
    signal = signal[first_index: ]

    sigmax = abs(max(signal))
    sigmin = abs(min(signal))
    
    return np.array(first_part + [
        v / sigmax if v > 0 else v / sigmin
        for v in signal
    ])


def chrominance(sig: str or np.array, settings: dict = CHROM_SETTINGS, 
                bounds: Tuple[int, int] = (0, -1), plot: bool = False):
    """"
    Apply the chrominance method to raw RGB data to extract and return
    a raw rPPG signal.
    
    Taken from "Robust Pulse Rate From Chrominance-Based rPPG" by de Haan and Jeanne
    """

    # make sure settings contain necessary info
    for key in CHROM_SETTINGS:
        if key not in settings:
            raise ValueError(f'Settings must contain value for key {key}.')

    # get raw RGB signals
    if isinstance(sig, str):
        r, g, b = _get_rgb_signals(sig, bounds)
    else:
        r = sig[:, 0]
        g = sig[:, 1]
        b = sig[:, 2]
    if plot:
        _plot_signals({'r': r, 'g': g, 'b': b}, 'Raw RGB Signals')
    
    # apply generic detrending and normalization to the raw signals
    r = detrend_w_poly(r)
    g = detrend_w_poly(g)
    b = detrend_w_poly(b)
    r = normalize_signal(r)
    g = normalize_signal(g)
    b = normalize_signal(b)
    if plot:
        _plot_signals({'r': r, 'g': g, 'b': b}, 'Detrended and Normalized RGB Signals')

    # normalize skin tones
    def _tonenorm(v):
        return v / np.sqrt(pow(r, 2) + pow(g, 2) + pow(b, 2))
    r_n, g_n, b_n = _tonenorm(r), _tonenorm(g), _tonenorm(b)
    if plot:
        _plot_signals({'r_n': r_n, 'g_n': g_n, 'b_n': b_n}, 'Normalized RGB Signals')

    # combine the terms
    xs = 3*r_n - 2*g_n
    ys = 1.5*r_n - g_n - 1.5*b_n
    
    # apply bandpass filter to each signal
    xf = bandpass(xs, settings['fr'], settings['freq'], settings['bandpass_order'])
    yf = bandpass(ys, settings['fr'], settings['freq'], settings['bandpass_order'])
    rf = bandpass(r_n, settings['fr'], settings['freq'], settings['bandpass_order'])
    gf = bandpass(g_n, settings['fr'], settings['freq'], settings['bandpass_order'])
    bf = bandpass(b_n, settings['fr'], settings['freq'], settings['bandpass_order'])

    # apply final transformation from the paper
    alpha = np.std(xf) / np.std(yf)
    signal = (3 * (1 - alpha / 2) * rf) - 2 * (1 + alpha / 2) * gf + ((3 * alpha / 2) * bf)

    return signal

def _plot_signals(signals: dict, title: str):
    """"
    Plot the signals in a dictionary.
    """

    for key in signals:
        plt.plot(signals[key], label = key)
    
    plt.title(title)
    plt.legend()
    plt.show()

def apply_wavelet(signal, wave='db2', level=1, cutoff_low=0.7, cutoff_high=3.0, fs=30):
    """
    Given a signal, apply wavelet transform to it and return a
    resulting signal.
    """

    # apply the wavelet transform, repeatedly according the the number of levels given
    filtered_signal = _wavelet_denoise(signal, wave, level)

    # interpolate the filtered signal to match the length of the original signal
    x_old = np.linspace(0, 1, len(filtered_signal))
    x_new = np.linspace(0, 1, len(signal))
    filtered_signal = np.interp(x_new, x_old, filtered_signal)

    # filter the interpolated signal to the desired frequency range
    b, a = _butter_bandpass(cutoff_low, cutoff_high, fs)
    filtered_signal = _filter_signal(filtered_signal, b, a)

    return filtered_signal


def _wavelet_denoise(signal, wavelet, level):
    
    # track signal at the end of each level
    vs = []
    sig = signal.copy()
    for _ in range(level):
        sig, cD = pywt.dwt(sig, wavelet)
        vs.append(sig)

    return vs[-1]


def _filter_signal(signal, b, a):
    """
    Filter a signal using a Butterworth filter with the given
    coefficients b and a.
    """
    filtered_signal = signal.copy()
    if len(b) == len(a) == 1:
        # if both b and a are of length 1, the filter is just a scalar multiplier
        filtered_signal = b * signal
    else:
        # apply the filter
        filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def _butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Create a Butterworth bandpass filter with the given parameters.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def get_peaks(signal: np.ndarray, fr: int, max_freq: float, peak_height: float,
    slice_filter_thresh: int, perc1: float, perc2: float, prominence: float or None = None,
    with_min_dist: bool = True, with_additional_filtering: bool = True):
    """
    The current standard method for peak detection. Includes the option to use the _filter_peaks
    method for even more aggressive filtering.
    """
    
    # skip any None values at the beggining of the signal
    first_index = None
    for i in range(len(signal)):
        if signal[i] is not None:
            first_index = i
            break
    signal = signal[first_index: ]
    
    # apply a min distance if its given, otherwise just make it 1
    if with_min_dist:
        min_dist = fr // max_freq
    else:
        min_dist = 1
    
    # if prominence is None, make it 0
    if prominence is None:
        prominence = 0
    
    peaks, _ = find_peaks(signal, height = peak_height, prominence = prominence, distance = min_dist)

    if with_additional_filtering:
        peaks = _filter_peaks(signal, peaks, fr, slice_filter_thresh, perc1, perc2)
    
    prominences = {p + first_index: prom for p, prom in zip(peaks, peak_prominences(signal, peaks)[0])}
    return [p + first_index for p in peaks], prominences


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
            # removed += len(to_remove)
    
    return peaks

def valley_filler(pv, fr, max_freq, peak_proms, valley_proms, prominence_threshold = 0.3):

    min_dist = 0.75 * (fr // max_freq)

    peaks = []
    i = 0
    while i < len(pv):

        # case where we are at the last peak in the sequence
        if i == len(pv) - 1:
            if pv[i][1] == 'p':
                peaks.append(pv[i][0])
                break
            elif pv[i][1] == 'v' and valley_proms[pv[i][0]] >= prominence_threshold:
                peaks.append(pv[i][0])
                break
    
        # case where the peak is a valley and the valley meets the prominence threshold
        if pv[i][1] == 'v' and valley_proms[pv[i][0]] >= prominence_threshold:
            peaks.append(pv[i][0])
            i += 1
            continue
        
        # case where the next peak is a peak (and not a valley), and thus no comparison needs to be done
        if pv[i + 1][1] == 'p':
            peaks.append(pv[i][0])
            i += 1
            continue

        # case where the next peak is a valley, and thus a comparison needs to be done
        else:
            if valley_proms[pv[i + 1][0]] >= prominence_threshold:

                if pv[i + 1][0] - pv[i][0] >= min_dist:
                    
                    peaks.append(pv[i][0])
                    peaks.append(pv[i + 1][0])
                    i += 2
                    continue
                
                else:
                    
                    if peak_proms[pv[i][0]] > valley_proms[pv[i + 1][0]]:
                        peaks.append(pv[i][0])
                        i += 2
                        continue
                    else:
                        peaks.append(pv[i + 1][0])
                        i += 2
                        continue
            
            else:
                peaks.append(pv[i][0])
                i += 2
                continue
            
    return peaks

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



if __name__ == '__main__':

    if len(sys.argv) == 3:
        subject = int(sys.argv[1])
        trial = int(sys.argv[2])
    else:
        subject = 5
        trial = 1

    truth = IeeeGroundTruth(subject, trial, directory = 'channel_data3')
    truth.align_rgb_bvp()

    # define sample signal interval of interest
    interval = 15 * truth.rgb_freq  # 20s times the frame rate from the video
    
    tracker = []
    for first_frame in range(0, len(truth.rgb), interval):

        rgb = truth.rgb[first_frame: first_frame + interval]
        # bvp has a higher sample rate so need to convert the rgb frames to the corresponding samples
        bvp = truth.bvp[truth.align_indices(first_frame): truth.align_indices(first_frame + interval)]

        signal = chrominance(rgb, plot = False)
        signal = apply_wavelet(signal, wave = 'db2', level = 1)
        signal = normalize_amplitude_to_1(signal)

        window = CHROM_SETTINGS['moving_avg_window']
        window = 5

        # apply the moving average filter
        orig = signal.copy()
        signal = n_moving_avg(signal, window)

        # add empty elements to the beginning of the array to maintain proper
        # positioning of the elements
        signal = [None]*((window - 1) // 2) + list(signal)
        signal = np.array(signal)

        bvp_window = 15
        bvp = n_moving_avg(bvp, bvp_window)
        # add empty elements to the beginning of the array to maintain proper
        # positioning of the elements
        bvp = [None]*((bvp_window - 1) // 2) + list(bvp)
        bvp = np.array(bvp)

        peaks, peak_proms = get_peaks(
            signal,
            CHROM_SETTINGS['fr'],
            CHROM_SETTINGS['freq'][1],
            # CHROM_SETTINGS['peak_height'],
            -1,
            CHROM_SETTINGS['slice_filter_thresh'],
            CHROM_SETTINGS['stringent_perc'],
            CHROM_SETTINGS['non_stringent_perc'],
            with_min_dist = True,
            with_additional_filtering = False,
            prominence = CHROM_SETTINGS['prominence']
        )

        signal_neg = [None if v is None else -v for v in signal]
        valleys, valley_proms = get_peaks(
                signal_neg,
                CHROM_SETTINGS['fr'],
                CHROM_SETTINGS['freq'][1],
        #         CHROM_SETTINGS['peak_height'],
                -1,
                CHROM_SETTINGS['slice_filter_thresh'],
                CHROM_SETTINGS['stringent_perc'],
                CHROM_SETTINGS['non_stringent_perc'],
                with_min_dist = True,
                with_additional_filtering = False,
                prominence = CHROM_SETTINGS['prominence']
        )

        # combine peaks and valleys
        peaks_valleys_comb = sorted(
            [(p, 'p') for p in peaks] + [(v, 'v') for v in valleys],
            key = lambda t: t[0]
        )

        peaks_enhanced = valley_filler(
            peaks_valleys_comb,
            CHROM_SETTINGS['fr'],
            CHROM_SETTINGS['freq'][1],
            peak_proms,
            valley_proms,
            prominence_threshold = 0.6
        )

        bvp_index = np.array(list(range(len(bvp)))) / 64
        signal_index = np.array(list(range(len(signal)))) / 30
        adjusted_peaks = [p / 30 for p in peaks]
        adjusted_valleys = [[v / 30 for v in valleys]]

        bvp = normalize_amplitude_to_1(bvp)
        signal = normalize_amplitude_to_1(signal)
        
        true_peaks, _ = get_peaks(
                bvp,
                CHROM_SETTINGS['fr'],
                CHROM_SETTINGS['freq'][1],
                0,
                CHROM_SETTINGS['slice_filter_thresh'],
                CHROM_SETTINGS['stringent_perc'],
                CHROM_SETTINGS['non_stringent_perc'],
                with_min_dist = False,
                with_additional_filtering = False
        )

        proms = peak_prominences(bvp, true_peaks)[0]
        true_peaks = [true_peaks[i] for i in range(len(true_peaks)) if proms[i] >= 0.6]


        true_ibis = get_ibis(true_peaks, fr = 64)
        peak_ibis = get_ibis(peaks, fr = 30)
        valley_ibis = get_ibis(valleys, fr = 30)
        ibis_peaks_valleys_comb = get_ibis(peaks_valleys_comb, CHROM_SETTINGS['fr'], with_valleys = True)
        ibis_peaks_enhanced = get_ibis(peaks_enhanced, CHROM_SETTINGS['fr'])

        tracker.append((get_hr(true_ibis), get_hr(ibis_peaks_enhanced)))
    
    
    for true, enh in tracker:
        print(enh - true)
    print(f'== Avg: {np.mean([abs(enh - true) for true, enh in tracker])} ==')
