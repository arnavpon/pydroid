import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from chrominance import chrominance, CHROM_SETTINGS as sett
from peaks import get_peaks
from signal_pross import (
    n_moving_avg, get_ibis, get_hr, 
    normalize_amplitude_to_1, get_hrv, 
    get_hr_from_fourier
)
from truth import IeeeGroundTruth
from wavelet import apply_wavelet

import sys
peak_height = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
print(f'Peak height: {peak_height}\n')


SETTINGS = {
    'fr': 30,  # frame rate
    'freq': (0.5, 3.34),  # bandpass frequency range
    'bandpass_order': 4,  # bandpass filter order
    'moving_avg_window': 6,  # moving average window size for smoothing
    'peak_height': peak_height,  # min peak height for peak detection
    'slice_filter_thresh': 2,  # min number of peaks allowed in a slice of the signal
    'stringent_perc': 85,  # more stringent percentile for peak filtering
    'non_stringent_perc': 75,  # less stringent percentile for peak filtering
}

    
def pipeline(sig: str or np.array, settings: dict = SETTINGS, wavelet = None, 
             with_valleys = False, with_fourier = False, bounds: Tuple[int, int] = (0, -1), plot: bool = False):

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
    
    if wavelet is not None:
        # len_before = len(signal)
        signal = apply_wavelet(signal, wave = 'db2', level = wavelet)
        # len_after = len(signal)
        # wavelet_mult = len_after / len_before
        if plot:
            plt.plot(signal)
            plt.title('Wavelet rPPG')
            plt.show()
    # else: wavelet_mult = 1
    signal = normalize_amplitude_to_1(signal)
    fhr = get_hr_from_fourier(signal, settings['fr'])

    # === Apply moving average to raw rPPG ===
    signal = n_moving_avg(signal, settings['moving_avg_window'])

    if plot:
        plt.plot(signal)
        plt.title('Smoothed rPPG')
        plt.show()

    # new_fr = int(settings['fr'] * wavelet_mult)
    # fhr = get_hr_from_fourier(signal, new_fr)

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
        with_additional_filtering = True
    )

    if with_valleys:
        valleys = get_peaks(
            -signal,
            settings['fr'],
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
    ibis = get_ibis(peaks, settings['fr'], with_valleys = with_valleys)
    hr = get_hr(ibis)

    hrv = get_hrv(ibis)
    # print(f'HR: {round(hr)}')

    if plot:
        plt.plot(ibis)
        plt.title('IBIs')
        plt.show()
    
    if with_fourier:
        return signal, fhr, hrv

    return signal, hr, hrv


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
    hrv = get_hrv(ibis)
    # print(f'HR: {round(hr)}')

    if plot:
        plt.plot(ibis)
        plt.title('IBIs')
        plt.show()
    
    return signal, hr, hrv


if __name__ == '__main__':

    raw_bvp_settings = sett.copy()

    S = 5
    E1=[]
    E2=[]
    E3=[]
    E4=[]
    for s in range(1, 8):
        if s in [4, 5]: continue
        print(f'Subject {s}:')

        truth = IeeeGroundTruth(s, 1)
        truth.align_rgb_bvp()
        raw_bvp_settings['fr'] = truth.bvp_freq

        interval = 20 * 30
        errs1 = []
        errs2 = []
        errs3 = []
        errs4 = []
        for i in range(1000, len(truth.rgb) - 1000, interval):
            # print(f'Starting at {i}:')
            sig_interval = [i, i + interval]
            truth_interval = [truth.align_indices(sig_interval[0]), truth.align_indices(sig_interval[1])]

            rgb = truth.rgb[sig_interval[0]: sig_interval[1], :]
            bvp = normalize_amplitude_to_1(truth.bvp[truth_interval[0]: truth_interval[1]])
            
            _, est_hr, est_hrv1 = pipeline(rgb, wavelet = None, with_valleys = False, plot = False)
            _, est_hr2, est_hrv2 = pipeline(rgb, wavelet = 1, with_valleys = False, plot = False)
            _, est_hr3, est_hrv3 = pipeline(rgb, wavelet = 2, with_valleys = False, plot = False)
            _, est_hr4, est_hrv4 = pipeline(rgb, wavelet = 3, with_valleys = False, plot = False)
            _, truth_hr, truth_hrv = pipeline_raw(bvp, settings = raw_bvp_settings, smoothing_window = 20, plot = False)
            
            # print('Estimated HR w/o fourier:', est_hr)
            # print('Estimated HR w/ fourier:', est_hr2)
            # print('Truth HRV:', truth_hr)
            # print()
            errs1.append(est_hr - truth_hr)
            errs2.append(est_hr2 - truth_hr)
            errs3.append(est_hr3 - truth_hr)
            errs4.append(est_hr4 - truth_hr)

        if len(errs1) > 0:
            print('Sum of error w/ Wavelet 0:', sum(errs1))
            print('Sum of error w/ Wavelet 1:', sum(errs2))
            print('Sum of error w/ Wavelet 2:', sum(errs3))
            print('Sum of error w/ Wavelet 3:', sum(errs4))
            errs1 = [abs(e) for e in errs1]
            errs2 = [abs(e) for e in errs2]
            errs3 = [abs(e) for e in errs3]
            errs4 = [abs(e) for e in errs4]
            print(f'Average error w/ Wavlet 0: {round(np.mean(errs1))} from {len(errs1)} samples')
            print(f'Average error w/ Wavlet 1: {round(np.mean(errs2))} from {len(errs2)} samples')
            print(f'Average error w/ Wavlet 2: {round(np.mean(errs3))} from {len(errs3)} samples')
            print(f'Average error w/ Wavlet 3: {round(np.mean(errs4))} from {len(errs4)} samples')
            print()

            E1.append(np.mean(errs1))
            E2.append(np.mean(errs2))
            E3.append(np.mean(errs3))
            E4.append(np.mean(errs4))

    print('MAINS:')
    print('Average error w/ Wavelet 0:', round(np.mean(E1)))
    print('Average error w/ Wavelet 1:', round(np.mean(E2)))
    print('Average error w/ Wavelet 2:', round(np.mean(E3)))
    print('Average error w/ Wavelet 3:', round(np.mean(E4)))