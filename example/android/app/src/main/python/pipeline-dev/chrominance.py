import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm

import signal_pross as sp
from pos import acquire_raw_signals


def chrominance(f, fr = 30,
                freq = (0.5, 3.34),
                moving_avg_window = 6,
                bandpass_order = 4,
                split_num = 4):

    df = pd.read_csv(f)
    r, g, b = df['r'].to_numpy(), df['g'].to_numpy(), df['b'].to_numpy()

    orig_len = len(r)
    nan_indices = [i for i, x in enumerate(r) if np.isnan(x)]
    r, g, b = r[~np.isnan(r)], g[~np.isnan(g)], b[~np.isnan(b)]
    
    if split_num is not None:
        quart = len(r) // split_num
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
    def _bandpass(v, order = bandpass_order):
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
    # return sp.n_moving_avg(signal, moving_avg_window)

    def _add_nans_back():
        new_signal = np.zeros(orig_len)

        i = 0
        j = 0
        while i < len(new_signal):
            
            val = signal[j]
            while i in nan_indices:
                new_signal[i] = np.nan
                i += 1
            
            new_signal[i] = val
            i += 1
            j += 1
        
        return new_signal

    return _add_nans_back()


def my_get_peaks(signal, fr = 30, freq = (0.5, 3.34), 
        peak_height = 0.00025, split = None,
        moving_avg_window = 6, bandpass_order = 4,
        slice_filter_thresh = 2, stringent_perc = 85,
        non_stringent_perc = 75, split_num = 4, plot = False):

    def _filter_peaks(peaks, perc1, perc2):
        peaks = np.array([p for p in peaks if signal[p] >= np.percentile(signal, perc2)])   
        
        # === peak walk ===
        removed = 0
        for i in range(0, len(signal) - fr, fr):
            j = i + fr
            slce = peaks[(peaks >= i) & (peaks < j)]
            if len(slce) > slice_filter_thresh:
                to_remove = [i for i in range(len(slce)) if signal[slce[i]] < np.percentile(signal[slce], perc1)]
                peaks = peaks[~np.isin(peaks, [slce[i] for i in to_remove])]
                removed += len(to_remove)
        #print('Removed with walk:', removed, 'peaks')
        # === end peak walk ===
        
        return peaks

    
    def _get_peaks(perc1, perc2, with_min_dist = True):
        if with_min_dist: min_dist = fr // freq[1]
        else: min_dist = 1
        peaks, _ = find_peaks(signal, height = peak_height, distance = min_dist)
        peaks = _filter_peaks(peaks, perc1, perc2)
        return peaks

    return _get_peaks(stringent_perc, non_stringent_perc, with_min_dist = True)


def get_signal_and_peaks(f, fr = 30, freq = (0.5, 3.34), 
        peak_height = 0.00025, split = None,
        moving_avg_window = 6, bandpass_order = 4,
        slice_filter_thresh = 2, stringent_perc = 85,
        non_stringent_perc = 75, split_num = 4, plot = False):

    signal = chrominance(f, fr, freq, moving_avg_window, bandpass_order, split_num)
    if signal is None:
        return None
    

    # === Peaks ===
    
    peaks = my_get_peaks(signal, fr, freq, slice_filter_thresh, peak_height, stringent_perc, non_stringent_perc)

    # === end peaks ===
    if plot:
        plot_thresh = 1000
        plt.plot(signal[0: plot_thresh])
        ps = peaks[peaks < plot_thresh]
        plt.scatter(ps, signal[ps], marker = 'x', color = 'red')
    
    return signal, peaks
    

def pipe(f, fr = 30, freq = (0.5, 3.34), 
        peak_height = 0.00025, split = None,
        moving_avg_window = 6, bandpass_order = 4,
        slice_filter_thresh = 2, stringent_perc = 85,
        non_stringent_perc = 75, split_num = 4, plot = False):
    
    _, peaks = get_signal_and_peaks(
        f, fr, freq, peak_height, split,
        moving_avg_window, bandpass_order,
        slice_filter_thresh, stringent_perc,
        non_stringent_perc, split_num, plot
    )

    ibis = sp.get_ibis(peaks)
    if split is None:
        return sp.get_hr(ibis)
    else:
        bin_size = len(ibis) // split
        print('Split results in bins of size:', bin_size)
        return [
            sp.get_hr(ibis[i * bin_size: (i + 1) * bin_size])
            for i in range(split)
        ]


def ieee_pipe(subj, trial, ground_truth_path = 'IBI', freq = (0.5, 3.0), peak_height = 0.00025, split = None,
        moving_avg_window = 6, bandpass_order = 4,
        slice_filter_thresh = 2, stringent_perc = 85,
        non_stringent_perc = 75, split_num = 4, verbose = True, plot = False):
    
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
    quart = len(gt) // split_num
    gt = gt[quart: -quart]
    
    # data path
    data_path = f'channel_data2/ieee-subject_{subj}-trial_{trial}-channel_data.csv'
    hr = pipe(
        data_path, freq = freq, split = split, peak_height = peak_height,
        moving_avg_window = moving_avg_window, bandpass_order = bandpass_order,
        slice_filter_thresh = slice_filter_thresh, stringent_perc = stringent_perc,
        non_stringent_perc = non_stringent_perc, split_num = split_num, plot = plot
    )

    # if verbose:
        # print(f'Subject {subj}, Trial {trial}:')
        # print(f'Ground truth HR from {ground_truth_path}: {np.mean(gt) if ground_truth_path == "HR" else 60 / np.mean(gt)}')
        # print(f'HR from chrominance: {hr}')
    if plot:
        plt.show()

    if split is None:
        return (np.mean(gt) if ground_truth_path == "HR" else 60 / np.mean(gt), hr)
    
    bin_size = len(gt) // split
    return ([
        np.median(gt[i * bin_size: (i + 1) * bin_size])
        if ground_truth_path == "HR"
        else 60 / np.median(gt[i * bin_size: (i + 1) * bin_size])
        for i in range(split)
    ], hr)


def test_variables(using, min_freqs, max_freqs, peak_heights,
    moving_avg_windows, bandpass_orders, slice_filter_threshs,
    stringent_percs, non_stringent_percs):

    # min_freqs = [0.5, 0.6, 0.8]
    # max_freqs = [2.5, 2.8, 3.0, 3.3, 3.7, 3.9]
    # peak_heights = [0.0008, 0.00012, 0.0003]
    # moving_avg_windows = [1, 5, 8]
    # bandpass_orders = [4]
    # slice_filter_threshs = [2]
    # stringent_percs = [80, 85, 90]
    # non_stringent_percs = [55, 65, 70]
    # test_variables(
    #     using, min_freqs, max_freqs, 
    #     peak_heights, moving_avg_windows,
    #     bandpass_orders, slice_filter_threshs,
    #     stringent_percs, non_stringent_percs
    # )

    print('Iterations:', len(min_freqs) * len(max_freqs) * len(peak_heights) * len(moving_avg_windows) * len(bandpass_orders) * len(slice_filter_threshs) * len(stringent_percs) * len(non_stringent_percs))

    params = []
    for i in tqdm(range(len(min_freqs))):
        min_freq = min_freqs[i]
        for j in tqdm(range(len(max_freqs))):
            max_freq = max_freqs[j]
            for peak_height in peak_heights:
                for moving_avg_window in moving_avg_windows:
                    for bandpass_order in bandpass_orders:
                        for slice_filter_thresh in slice_filter_threshs:
                            for stringent_perc in stringent_percs:
                                for non_stringent_perc in non_stringent_percs:

                                    errs = []
                                    for subj in using:
                                        for trial in using[subj]:
                                            gt, hr = ieee_pipe(subj, trial, ground_truth_path = 'HR', freq = (min_freq, max_freq), peak_height = peak_height, plot = False,
                                                moving_avg_window = moving_avg_window, bandpass_order = bandpass_order,
                                                slice_filter_thresh = slice_filter_thresh, stringent_perc = stringent_perc,
                                                non_stringent_perc = non_stringent_perc, verbose = False)
                                            err = abs(gt - hr)
                                            errs.append((err, pow(err, 2)))
                                    
                                    me = np.mean([e[0] for e in errs])
                                    mse = np.mean([e[1] for e in errs])
                                    params.append({
                                        'mean_err': me,
                                        'mse': mse,
                                        'min_freq': min_freq,
                                        'max_freq': max_freq,
                                        'peak_height': peak_height,
                                        'moving_avg_window': moving_avg_window,
                                        'bandpass_order': bandpass_order,
                                        'slice_filter_thresh': slice_filter_thresh,
                                        'stringent_perc': stringent_perc,
                                        'non_stringent_perc': non_stringent_perc,
                                    })

                                    # print(f'Params: Min Freq: {min_freq}; Max Freq: {max_freq}; Peak Height: {peak_height}; Moving Avg Window: {moving_avg_window}; Bandpass Order: {bandpass_order}; Slice Filter Thresh: {slice_filter_thresh}; Stringent Perc: {stringent_perc}; Non-Stringent Perc: {non_stringent_perc}')
                                    # print(f'Error: {err}%')
                                    # print()

    params.sort(key = lambda x: x['mse'])
    for p in params:
        print(p)
        print()


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
            gt, hr = ieee_pipe(
                subj, trial, split = 12,
                ground_truth_path = 'HR',
                freq = (0.5, 3.166),
                peak_height = 0.00012,
                moving_avg_window = 8,
                bandpass_order = 4,
                slice_filter_thresh = 2,
                stringent_perc = 80,
                non_stringent_perc = 55,
                plot = True
            )

            print(f'Subject {subj}, Trial {trial}:')
            print(f'Ground truth HR from HR: {np.median(gt)}')# : {gt}')
            print(f'HR from chrominance: {np.median(hr)}')# : {hr}')
            
            gt_reported = gt if isinstance(gt, float) else np.mean(gt)
            hr_reported = hr if isinstance(hr, float) else np.mean(hr)
            err = round(abs(gt_reported - hr_reported) / gt_reported * 100, 2)
            errs.append(err)
            print(f'Error: {err}%')
            print()
    
    print(f'Average error: {np.mean(errs)}%')


    # {'mean_err': 9.420449004715689, 'mse': 150.6233113389274, 'min_freq': 0.5, 'max_freq': 3.7, 'peak_height': 0.00012, 'moving_avg_window': 8, 'bandpass_order': 4, 'slice_filter_thresh': 2, 'stringent_perc': 80, 'non_stringent_perc': 55}

    # {'mean_err': 9.420449004715689, 'mse': 150.6233113389274, 'min_freq': 0.5, 'max_freq': 3.7, 'peak_height': 0.00012, 'moving_avg_window': 8, 'bandpass_order': 4, 'slice_filter_thresh': 2, 'stringent_perc': 85, 'non_stringent_perc': 55}

    # {'mean_err': 9.420449004715689, 'mse': 150.6233113389274, 'min_freq': 0.5, 'max_freq': 3.7, 'peak_height': 0.00012, 'moving_avg_window': 8, 'bandpass_order': 4, 'slice_filter_thresh': 2, 'stringent_perc': 90, 'non_stringent_perc': 55}

    # {'mean_err': 9.2411477407509, 'mse': 151.22351673537202, 'min_freq': 0.5, 'max_freq': 3.9, 'peak_height': 0.00012, 'moving_avg_window': 8, 'bandpass_order': 4, 'slice_filter_thresh': 2, 'stringent_perc': 80, 'non_stringent_perc': 55}

    # {'mean_err': 9.2411477407509, 'mse': 151.22351673537202, 'min_freq': 0.5, 'max_freq': 3.9, 'peak_height': 0.00012, 'moving_avg_window': 8, 'bandpass_order': 4, 'slice_filter_thresh': 2, 'stringent_perc': 85, 'non_stringent_perc': 55}
