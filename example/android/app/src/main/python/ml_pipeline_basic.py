import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, resample, filtfilt, find_peaks, peak_prominences
from sklearn.decomposition import FastICA
import pywt
from typing import Tuple
import xgboost as xgb
import os

mod = xgb.Booster()



def pipeline(boxes, frames):

    rgb = _get_rgb_from_boxes(boxes, frames)
    print('got here now')
    rgb = process_rgb(rgb)
    print('got thru')
    data = prepare_data_for_ml(rgb, num_feats_per_channel = 8, skip_amount = 12)
    print(f'got data with shape: {data.shape}')

    print('loading lgb model')
    models_path, _ = os.path.split(os.path.realpath(__file__))
    print(models_path)
    model_file = os.path.join(models_path,'lonePineGBM.xgb')
    mod.load_model(model_file)
    print('...loaded')

    print('conveting to dmatrix')
    dmatrix = xgb.DMatrix(data)
    print('...converted')
    print('predicting')
    preds = mod.predict(dmatrix)
    print('...predicted')

    print('prediction shape')
    print(preds.shape)
    print(list(preds))

    print('getting heart rate:')
    res =  get_heart_rate(preds)
    print(f'...got it: {res}')
    return res
    

def _get_rgb_from_boxes(boxes, frames, roi_percentage=0.5):

    channel_data = {
        'r': [],
        'g': [],
        'b': []
    }
    for bbox, frame in zip(boxes, frames):
        print(f'bbox: {bbox}')
        roi_width = bbox['x2'] - bbox['x1']
        roi_height = bbox['y2'] - bbox['y1']

        xi = int(max(0, int(bbox['x1'] + ((roi_width * (1 - roi_percentage)) / 2))))
        xw = int(roi_width * roi_percentage)

        yi = int(max(0, bbox['y1']))
        yh = int(roi_height * roi_percentage)

        print(f'got here with: {xi}, {xw}, {yi}, {yh}')
        cropped_img = frame[yi: yi + yh, xi: xi + xw]

        channels = cv2.split(cropped_img)
        for k, c in zip(channel_data.keys(), channels):
            channel_data[k].append(np.mean(c))
    
    return pd.DataFrame(channel_data).to_numpy()


def process_rgb(rgb):

    for i in range(rgb.shape[1]):
        rgb[:, i] = detrend_w_poly(rgb[:, i])
        rgb[:, i] = apply_wavelet(
            rgb[:, i],
            cutoff_low = 0.5,
            cutoff_high = 3,
            wave = 'db2', 
            level = 1
        )
        rgb[:, i] = min_max_scale(rgb[:, i])
    
    return rgb
    
def prepare_data_for_ml(rgb, num_feats_per_channel = 5, skip_amount = 10):
    
    new_len = int((64 / 30) * len(rgb))
    rgb_upsampled = np.zeros((new_len, rgb.shape[1]))
    for col in range(rgb_upsampled.shape[1]):
        rgb_upsampled[:, col] = upsample_signal(
            rgb[:, col],
            new_len,
            old_fs = 30,
            new_fs = 64
        )
    
    rgb_vel = np.zeros((rgb_upsampled.shape[0] - 2, rgb_upsampled.shape[1]))
    rgb_acc = np.zeros((rgb_upsampled.shape[0] - 2, rgb_upsampled.shape[1]))
    for col in range(rgb_vel.shape[1]):

        vel = np.diff(rgb_upsampled[:, col], n = 1)[1: ]
        acc = np.diff(rgb_upsampled[:, col], n = 2)
        rgb_vel[:, col] = vel
        rgb_acc[:, col] = acc

    rgb_upsampled = rgb_upsampled[2: , :]
    chrom = prepare_chrominance_as_feature(rgb_upsampled)

    # get ica feature
    ica_feat = perform_ica(rgb_upsampled)

    # add "memory" features
    mems = get_memory_features(rgb_upsampled, num_feats_per_channel, skip_amount)
    rgb_upsampled = rgb_upsampled[num_feats_per_channel * skip_amount: , :]
    chrom = chrom[num_feats_per_channel * skip_amount: ]
    rgb_vel = rgb_vel[num_feats_per_channel * skip_amount: , :]
    rgb_acc = rgb_acc[num_feats_per_channel * skip_amount: , :]
    ica_feat = ica_feat[num_feats_per_channel * skip_amount: ]


    data = {
        'chrom': chrom,
        'ica_feat': ica_feat,
        'r': rgb_upsampled[:, 0],
        'g': rgb_upsampled[:, 1],
        'b': rgb_upsampled[:, 2],
        'r_vel': rgb_vel[:, 0],
        'g_vel': rgb_vel[:, 1],
        'b_vel': rgb_vel[:, 2],
        'r_acc': rgb_acc[:, 0],
        'g_acc': rgb_acc[:, 1],
        'b_acc': rgb_acc[:, 2],
    }
    for k, v in mems.items():
        data[k] = v

    print(f'we have the keys and there are {len(list(data.keys()))}:', data.keys())
    return pd.DataFrame(data).to_numpy()

def get_memory_features(rgb_upsampled, num_feats_per_channel, skip_amount):

    channels = ['r', 'g', 'b']
    mems = {f'{c}_mem_{i}': [] for i in range(num_feats_per_channel) for c in channels}
    
    for idx in range(num_feats_per_channel * skip_amount, rgb_upsampled.shape[0]):
        for i, c in enumerate(channels):
            for j in range(num_feats_per_channel):
                mems[f'{c}_mem_{j}'].append(rgb_upsampled[idx - (j * skip_amount), i])
    
    return mems

def upsample_signal(signal, new_length, old_fs = 30, new_fs = 64):
    f = resample(signal, new_length)
    return f

def prepare_chrominance_as_feature(rgb):
    chrom = chrominance(rgb, CHROM_SETTINGS, None, False)
    chrom = apply_wavelet(chrom, 'db2', 2)
    chrom = bandpass(chrom, 64, [0.67, 3.0], 4)
    chrom = min_max_scale(chrom)
    return chrom

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

CHROM_SETTINGS = {
    'fr': 30,  # frame rate
    'freq': (0.67, 3.0),  # bandpass frequency range
    'bandpass_order': 4,  # bandpass filter order
    'moving_avg_window': 5,  # moving average window size for smoothing
    'peak_height': 0.00025,  # min peak height for peak detection
    'slice_filter_thresh': 2,  # min number of peaks allowed in a slice of the signal
    'stringent_perc': 85,  # more stringent percentile for peak filtering
    'non_stringent_perc': 75,  # less stringent percentile for peak filtering
}


def chrominance(sig: str or np.array, settings: dict = CHROM_SETTINGS, bounds: Tuple[int, int] = (0, -1), plot: bool = False):
    """"
    Apply the chrominance method to raw RGB data to extract and return
    a raw rPPG signal.
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
    # if plot:
    #     _plot_signals({'r': r, 'g': g, 'b': b}, 'Raw RGB Signals')
    
    # apply generic detrending and normalization to the raw signals
    r = detrend_w_poly(r)
    g = detrend_w_poly(g)
    b = detrend_w_poly(b)
    r = normalize_signal(r)
    g = normalize_signal(g)
    b = normalize_signal(b)
    # if plot:
    #     _plot_signals({'r': r, 'g': g, 'b': b}, 'Detrended and Normalized RGB Signals')

    # normalize skin tones
    def _tonenorm(v):
        return v / np.sqrt(pow(r, 2) + pow(g, 2) + pow(b, 2))
    r_n, g_n, b_n = _tonenorm(r), _tonenorm(g), _tonenorm(b)
    # if plot:
    #     _plot_signals({'r_n': r_n, 'g_n': g_n, 'b_n': b_n}, 'Normalized RGB Signals')

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


def _get_rgb_signals(path: str, bounds: Tuple[int, int]):
    """"
    Get the raw RGB signals from a CSV file.
    """

    df = pd.read_csv(path)
    for col in ['r', 'g', 'b']:
        if col not in df.columns:
            raise ValueError(f'Column {col} not found in CSV file.')
    
    r = df['r'].to_numpy()[bounds[0]: bounds[1]]
    g = df['g'].to_numpy()[bounds[0]: bounds[1]]
    b = df['b'].to_numpy()[bounds[0]: bounds[1]]
    return r, g, b

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

    # # filter the interpolated signal to the desired frequency range
    # b, a = _butter_bandpass(cutoff_low, cutoff_high, fs)
    # filtered_signal = _filter_signal(filtered_signal, b, a)

    return filtered_signal


def _wavelet_denoise(signal, wavelet, level):
    
    # track signal at the end of each level
    vs = []
    sig = signal.copy()
    for _ in range(level):
        sig, cD = pywt.dwt(sig, wavelet)
        vs.append(sig)

    return vs[-1]

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


def get_heart_rate(y_pred):
    
    y_pred = process_signal(y_pred)
    num_cycles = len(y_pred) // 960
    
    if num_cycles == 0:
        cycles = [y_pred]
    else:
        cycles = []
        for i in range(num_cycles):
            cycles.append(y_pred[i*960: (i+1)*960])

    res = []
    for cyc in cycles:

        pred_peaks, _ = get_peaks_v2(cyc, 64, 3.0, -1, prominence = 0.17, with_min_dist = True, with_valleys = False)

        if len(pred_peaks) >= 2:
            pred_ibis = np.diff(pred_peaks) / 64
            pred_hr = 60 / np.mean(pred_ibis)
            pred_hrv = np.sqrt(np.mean(np.square(np.diff(pred_ibis))))
        else:
            pred_hr = 0
            pred_hrv = 0

        res.append((pred_hr, pred_hrv))
    

    return res

def get_peaks_v2(signal: np.ndarray, fr: int, max_freq: float, peak_height: float,
    prominence: float = 0, with_valleys: bool = False,
    with_min_dist: bool = True):
    
    # skip any None values at the beggining of the signal
    first_index = None
    for i in range(len(signal)):
        if signal[i] is not None:
            first_index = i
            break
    signal = signal[first_index: ]

    def _peak_getter(sig):
        min_dist = fr // max_freq if with_min_dist else 1
        
        peaks, _ = find_peaks(sig, height = peak_height, prominence = prominence, distance = min_dist)
        
        prominences = {p + first_index: prom for p, prom in zip(peaks, peak_prominences(sig, peaks)[0])}
        return [p + first_index for p in peaks], prominences

    peaks, peak_proms = _peak_getter(signal)
    if with_valleys:
        valleys, valley_prominences = _peak_getter(-signal)

        peaks = sorted(list(set(peaks + valleys)))
        peak_proms = {**peak_proms, **valley_prominences}
        return peaks, peak_proms

    return np.array(peaks), peak_proms


def process_signal(y_pred, smoothing_window = 10, use_bandpass = True):
    
    orig_len = len(y_pred)
    y_pred = n_moving_avg(y_pred, smoothing_window)
    y_pred = resample(y_pred, orig_len)

    if use_bandpass:
        y_pred = bandpass(y_pred, 64, [0.7, 4.0], 4)
    y_pred = min_max_scale(y_pred)

    return y_pred


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