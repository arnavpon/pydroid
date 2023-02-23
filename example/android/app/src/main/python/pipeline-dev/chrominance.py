import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

import signal_pross as sp
from pos import acquire_raw_signals


def chrominance(f, fr = 30, freq = (0.5, 3.34), moving_avg_window = 6):

    df = pd.read_csv(f)
    r, g, b = df['r'].to_numpy(), df['g'].to_numpy(), df['b'].to_numpy()
    
    quart = len(r) // 4
    r, g, b = r[quart: -quart], g[quart: -quart], b[quart: -quart]

    # === skin tone normalization ===
    def tonenorm(v):
        return v / np.sqrt(pow(r, 2) + pow(g, 2) + pow(b, 2))
    rn, gn, bn = tonenorm(r), tonenorm(g), tonenorm(b)
    # === end skin tone normalization ===

    # === hardcoded constants from the paper ===
    rs = 0.7681 * rn
    gs = 0.5121 * gn
    bs = 0.3841 * bn
    # === end hardcoded constants from the paper ===

    # === combining the terms ===
    xs = 3*rn - 2*gn
    ys = 1.5*rn - gn - 1.5*bn
    # === end combining the terms ===

    # === bandpass again ===
    def _bandpass(v, order = 4):
        # return np.convolve(v, np.hamming(int(fr / low)))[::int(fr / high)]
        nyquist_freq = 0.5 * fr
        low = freq[0] / nyquist_freq
        high = freq[1] / nyquist_freq
        b, a = butter(order, [low, high], btype = 'band')
        filtered = filtfilt(b, a, v)
        return filtered
    
    xf = _bandpass(xs)
    yf = _bandpass(ys)
    rf = _bandpass(rn)
    gf = _bandpass(gn)
    bf = _bandpass(bn)
    # === end bandpass ===

    alpha = np.std(xf) / np.std(yf)
    signal = (3*(1 - alpha/2)*rf) - 2*(1 + alpha/2)*gf + ((3*alpha/2)*bf)

    def _normalize_signal(signal, target_amplitude):
        max_amplitude = np.max(np.abs(signal))
        scale_factor = target_amplitude / max_amplitude
        normalized_signal = signal * scale_factor
        return normalized_signal

    #return _normalize_signal(signal, 0.001)
    return sp.n_moving_avg(signal, moving_avg_window)

def pipe(f, fr = 30, freq = (0.5, 3.34), peak_height = 0.00025, split = None):

    signal = chrominance(f, fr, freq)
    if signal is None:
        return None
    

    # === Peaks ===
    def _filter_peaks(peaks, perc1, perc2):
        peaks = np.array([p for p in peaks if signal[p] >= np.percentile(signal, perc2)])   
        
        # === peak walk ===
        removed = 0
        for i in range(0, len(signal) - fr, fr):
            j = i + fr
            slce = peaks[(peaks >= i) & (peaks < j)]
            if len(slce) > 2:
                to_remove = [i for i in range(len(slce)) if signal[slce[i]] < np.percentile(signal[slce], perc1)]
                peaks = peaks[~np.isin(peaks, [slce[i] for i in to_remove])]
                removed += len(to_remove)
        print('Removed with walk:', removed, 'peaks')
        # === end peak walk ===
        
        return peaks

    
    def _get_peaks(perc1, perc2, with_min_dist = True):

        if with_min_dist: min_dist = fr // freq[1]
        else: min_dist = 1
        peaks, _ = find_peaks(signal, height = peak_height, distance = min_dist)
        peaks = _filter_peaks(peaks, perc1, perc2)
        return peaks
    peaks = _get_peaks(85, 75, with_min_dist = True)

    # === end peaks ===
    plot_thresh = 1000
    plt.plot(signal[0: plot_thresh])
    ps = peaks[peaks < plot_thresh]
    plt.scatter(ps, signal[ps], marker = 'x', color = 'red')
    
    ibis = sp.get_ibis(peaks)

    if split is None:
        return sp.get_hr(ibis)
    else:
        bin_size = len(ibis) // split
        return [
            sp.get_hr(ibis[i * bin_size: (i + 1) * bin_size])
            for i in range(split)
        ]

def ieee_pipe(subj, trial, ground_truth_path = 'IBI', peak_height = 0.00025, plot = False):
    
    # get ground truth
    gt_path = f'validation_data/IEEE_data/subject_{subj}/trial_{trial}/empatica_e4/{ground_truth_path}.csv'
    if ground_truth_path == 'HR':
        gt = pd.read_csv(gt_path)
        gt = list(gt[list(gt.columns)[0]])[1: ]
    elif ground_truth_path == 'IBI':
        gt = pd.read_csv(gt_path)
        gt = list(gt[list(gt.columns)[1]])
    else:
        raise Exception('Invalid ground truth arg')

    # === For lack of a better idea, we use the middle 50% of ground truth and measured
    # signal, even though they don't actually line up properly. ===
    quart = len(gt) // 4
    gt = gt[quart: -quart]
    
    # data path
    data_path = f'channel_data2/ieee-subject_{subj}-trial_{trial}-channel_data.csv'
    hr = pipe(data_path, freq = (0.5, 3.0), split = None, peak_height = peak_height)

    print(f'Subject {subj}, Trial {trial}:')
    print(f'Ground truth HR from {ground_truth_path}: {np.mean(gt) if ground_truth_path == "HR" else 60 / np.mean(gt)}')
    print(f'HR from chrominance: {hr}')
    if plot:
        plt.show()

    return (np.mean(gt) if ground_truth_path == "HR" else 60 / np.mean(gt), hr)


if __name__ == '__main__':
    
    using = {
        '001': ['001'],
        '002': ['001'],
        '003': ['001'],
        '004': ['001'],
        '005': ['001'],
        '006': ['001'],
        '007': ['001'],
    }

    errs = []
    for subj in using:
        for trial in using[subj]:
            gt, hr = ieee_pipe(subj, trial, ground_truth_path = 'HR', peak_height = 0.00012, plot = True)
            err = round(abs(gt - hr) / gt * 100, 2)
            errs.append(err)
            print(f'Error: {err}%')
            print()
    
    print(f'Average error: {np.mean(errs)}%')
