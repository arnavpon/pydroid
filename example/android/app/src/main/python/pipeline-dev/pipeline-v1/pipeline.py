import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from chrominance import chrominance, CHROM_SETTINGS as sett
from peaks import get_peaks
from signal_pross import n_moving_avg, get_ibis, get_hr, normalize_amplitude_to_1
from truth import IeeeGroundTruth
from wavelet import apply_wavelet

import sys
peak_height = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
print(f'Peak height: {peak_height}\n')


SETTINGS = {
    'fr': 30,  # frame rate
    'freq': (0.5, 3.34),  # bandpass frequency range
    'bandpass_order': 4,  # bandpass filter order
    'moving_avg_window': 5,  # moving average window size for smoothing
    'peak_height': peak_height,  # min peak height for peak detection
    'slice_filter_thresh': 2,  # min number of peaks allowed in a slice of the signal
    'stringent_perc': 85,  # more stringent percentile for peak filtering
    'non_stringent_perc': 75,  # less stringent percentile for peak filtering
}

    
def pipeline(sig: str or np.array, settings: dict = SETTINGS, with_wavelet = False, with_valleys = False, bounds: Tuple[int, int] = (0, -1), plot: bool = False):

    if isinstance(sig, str):
        signal = chrominance(
            sig,
            bounds = bounds,
            plot = False
        )
    elif sig.shape[1] == 3:
        signal = chrominance(sig)
    else:
        signal = sig

    if plot:
        plt.plot(signal)
        plt.title('Raw rPPG')
        plt.show()

    # === Apply moving average to raw rPPG ===
    signal = n_moving_avg(signal, settings['moving_avg_window'])

    if plot:
        plt.plot(signal)
        plt.title('Smoothed rPPG')
        plt.show()
    
    if with_wavelet:
        len_before = len(signal)
        signal = apply_wavelet(signal, wave = 'db2', level = 1)
        len_after = len(signal)
        wavelet_mult = len_after / len_before
        if plot:
            plt.plot(signal)
            plt.title('Wavelet rPPG')
            plt.show()
    else: wavelet_mult = 1

    new_fr = int(settings['fr'] * wavelet_mult)

    signal = normalize_amplitude_to_1(signal)

    # === Get peaks from smoothed rPPG ===
    peaks = get_peaks(
        signal,
        new_fr,
        settings['freq'][1],
        settings['peak_height'],
        settings['slice_filter_thresh'],
        settings['stringent_perc'],
        settings['non_stringent_perc'],
        with_min_dist = True,
        with_additional_filtering = True
    )

    if with_valleys:
        valleys = get_peaks(
            -signal,
            new_fr,
            settings['freq'][1],
            settings['peak_height'],
            settings['slice_filter_thresh'],
            settings['stringent_perc'],
            settings['non_stringent_perc'],
            with_min_dist = True,
            with_additional_filtering = True
        )

        peaks = sorted(
            [(p, 'p') for p in peaks] + [(v, 'v') for v in valleys],
            key = lambda t: t[0]
        )

    if plot:
        plt.plot(signal)
        plt.scatter([p[0] for p in peaks], [signal[p[0]] for p in peaks], marker = 'x', c = 'r')
        plt.title('Peaks')
        plt.show()

    # === Get IBI and HR from peaks ===
    ibis = get_ibis(peaks, new_fr, with_valleys = with_valleys)
    hr = get_hr(ibis)
    # print(f'HR: {round(hr)}')

    if plot:
        plt.plot(ibis)
        plt.title('IBIs')
        plt.show()
    
    return signal, hr


def pipeline_raw(signal, smoothing_window = None, settings: dict = sett, plot: bool = False):

    signal = signal.flatten()
    if smoothing_window is not None:
        signal = n_moving_avg(signal, window = smoothing_window)

    # === Get peaks from smoothed rPPG ===
    peaks = get_peaks(
        signal,
        settings['fr'],
        settings['freq'][1],
        settings['peak_height'],
        settings['slice_filter_thresh'],
        settings['stringent_perc'],
        settings['non_stringent_perc'],
        with_min_dist = True,
        with_additional_filtering = False
    )

    if plot:
        plt.plot(signal)
        plt.scatter(peaks, [signal[p] for p in peaks], marker = 'x', c = 'r')
        plt.title('Peaks')
        plt.show()

    # === Get IBI and HR from peaks ===
    ibis = get_ibis(peaks, fr = settings['fr'])
    hr = get_hr(ibis)
    # print(f'HR: {round(hr)}')

    if plot:
        plt.plot(ibis)
        plt.title('IBIs')
        plt.show()
    
    return signal, hr


if __name__ == '__main__':

    raw_bvp_settings = sett.copy()

    S = 5
    E1=[]
    E2=[]
    for s in range(1, 8):
        print(f'Subject {s}:')

        truth = IeeeGroundTruth(s, 1)
        truth.align_rgb_bvp()
        raw_bvp_settings['fr'] = truth.bvp_freq

        interval = 20 * 30
        errs1 = []
        errs2 = []
        for i in range(1000, len(truth.rgb) - 1000, interval):
            # print(f'Starting at {i}:')
            sig_interval = [i, i + interval]
            truth_interval = [truth.align_indices(sig_interval[0]), truth.align_indices(sig_interval[1])]

            rgb = truth.rgb[sig_interval[0]: sig_interval[1], :]
            bvp = normalize_amplitude_to_1(truth.bvp[truth_interval[0]: truth_interval[1]])
            
            _, est_hr = pipeline(rgb, with_wavelet = True, plot = False)
            _, est_hr2 = pipeline(rgb, with_wavelet = True, with_valleys = True, plot = False)
            _, truth_hr = pipeline_raw(bvp, settings = raw_bvp_settings, smoothing_window = 20, plot = False)
            
            # print('Estimated HR:', est_hr)
            # print('Truth HR:', truth_hr)
            # print()
            errs1.append(est_hr - truth_hr)
            errs2.append(est_hr2 - truth_hr)

        if len(errs1) > 0:
            print('Sum of the w/o valleys error:', sum(errs1))
            print('Sum of the w/ valleys error:', sum(errs2))
            errs1 = [abs(e) for e in errs1]
            errs2 = [abs(e) for e in errs2]
            print(f'Average error w/o valleys: {round(np.mean(errs1))} from {len(errs1)} samples')
            print(f'Average error w/ valleys: {round(np.mean(errs2))} from {len(errs2)} samples')
            print()

            E1.append(np.mean(errs1))
            E2.append(np.mean(errs2))

    print('MAINS:')
    print('Average error w/o valleys:', round(np.mean(E1)))
    print('Average error w/ valleys:', round(np.mean(E2)))
