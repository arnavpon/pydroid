import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from chrominance import chrominance, CHROM_SETTINGS as sett
from peaks import get_peaks
from signal_pross import n_moving_avg, get_ibis, get_hr, normalize_to_amplitude
from truth import IeeeGroundTruth


SETTINGS =CHROM_SETTINGS = {
    'fr': 30,  # frame rate
    'freq': (0.5, 3.34),  # bandpass frequency range
    'bandpass_order': 4,  # bandpass filter order
    'moving_avg_window': 5,  # moving average window size for smoothing
    'peak_height': 0.00025,  # min peak height for peak detection
    'slice_filter_thresh': 2,  # min number of peaks allowed in a slice of the signal
    'stringent_perc': 85,  # more stringent percentile for peak filtering
    'non_stringent_perc': 75,  # less stringent percentile for peak filtering
}

    
def pipeline(sig: str or np.array, settings: dict = sett, bounds: Tuple[int, int] = (0, -1), plot: bool = False):

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
    signal = n_moving_avg(signal, window = 10)

    if plot:
        plt.plot(signal)
        plt.title('Smoothed rPPG')
        plt.show()

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
    ibis = get_ibis(peaks, settings['fr'])
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

    for s in range(1, 8):
        print(f'Subject {s}:')

        truth = IeeeGroundTruth(s, 1)
        truth.align_rgb_bvp()
        raw_bvp_settings['fr'] = truth.bvp_freq

        interval = 15 * 30
        errs = []
        for i in range(1000, len(truth.rgb) - 1000, interval):
            try:
                # print(f'Starting at {i}:')
                sig_interval = [i, i + interval]
                truth_interval = [truth.align_indices(sig_interval[0]), truth.align_indices(sig_interval[1])]

                rgb = truth.rgb[sig_interval[0]: sig_interval[1], :]
                bvp = normalize_to_amplitude(truth.bvp[truth_interval[0]: truth_interval[1]], 1)
                signal, est_hr = pipeline(rgb, plot = False)
                truth_signal, truth_hr = pipeline_raw(bvp, settings = raw_bvp_settings, smoothing_window = 20, plot = False)
                
                # print('Estimated HR:', est_hr)
                # print('Truth HR:', truth_hr)
                # print()
                errs.append(est_hr - truth_hr)

            except Exception as e: print('Error', e)

        if len(errs) > 0:
            print('Sum of the error:', sum(errs))
            errs = [abs(e) for e in errs]
            print(f'Average error: {round(np.mean(errs))} from {len(errs)} samples')
            print()
