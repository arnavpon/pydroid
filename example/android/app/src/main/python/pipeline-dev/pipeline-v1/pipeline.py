import matplotlib.pyplot as plt
from typing import Tuple

from chrominance import chrominance, CHROM_SETTINGS as sett
from peaks import get_peaks
from signal_pross import n_moving_avg, get_ibis, get_hr

    
def pipeline(path: str, settings: dict = sett, bounds: Tuple[int, int] = (0, -1), plot: bool = False):

    # === Get raw rPPG signal using Chrominance ===
    signal = chrominance(
        path,
        bounds = (START_POINT, END_POINT),
        plot = False
    )

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
    print(f'HR: {round(hr)}')

    if plot:
        plt.plot(ibis)
        plt.title('IBIs')
        plt.show()


if __name__ == '__main__':

    START_POINT = 1500
    END_POINT = 1900
    PATH = 'channel_data/ieee-subject-002-trial-001.csv'

    for s in range(1, 8):
        print('On subject:', s)
        path = f'channel_data/ieee-subject-00{s}-trial-001.csv'
        pipeline(path, plot = True if s == 7 else False)
        print()
