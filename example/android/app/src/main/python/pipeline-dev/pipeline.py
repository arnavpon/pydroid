"""
Module handling the calculation of HR and HRV using color channel vectors
derived from a face video.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import signal_pross as sp
from tracking import DEFAULT_CSV_NAME


def load_channels(path = DEFAULT_CSV_NAME):
    """
    Load ROI channel vectors from local csv file.
    """
    
    channels = pd.read_csv(path)
    return {
        k: channels[k].to_numpy()
        for k in channels.columns
    }


def pipeline(path = DEFAULT_CSV_NAME, 
            moving_average_window = 15,
            lim = 0, with_plots = True):

    # === STEP: Load color signals
    channels = load_channels(path = path)

    if with_plots:
        plt.title('Raw Channels')
        for k in channels: plt.plot(channels[k], label = k)
        plt.legend()
        plt.show()

    # === STEP: Detrend each channel
    channels = {
        k: sp.detrend_w_poly(channels[k][lim: ])
        for k in channels
    }

    # === STEP: Normalize each channel
    channels = {
        k: sp.normalize_detrended(channels[k])
        for k in channels
    }

    if with_plots:
        plt.title('Normalized Detrended Channels')
        for k in channels: plt.plot(channels[k], label = k)
        plt.legend()
        plt.show()

    # === STEP: Apply ICA and get component with highest spectrum peak
    # NOTE: For now, we use JADE
    X = np.stack([channels[k] for k in channels], axis = 1)
    bvp_comp = sp.get_bvp_w_ica(X, ica_method = sp.jade_v4)

    if with_plots:
        plt.title(f'ICA Selected Component')
        plt.plot(bvp_comp)
        plt.show()

    # === STEP: Apply N-point moving average filter to the peak comp
    fcomp = sp.n_moving_avg(bvp_comp, window = moving_average_window)
    
    if with_plots:
        plt.title(f'{moving_average_window}-point Moving Average Filtered Component')
        plt.plot(fcomp)
        plt.show()

    # === STEP: Get peaks from filtered component
    peaks = sp.get_peaks(fcomp)

    if with_plots:
        plt.plot(fcomp)
        plt.scatter(peaks, [fcomp[i] for i in peaks], marker = 'x', color = 'red')
        plt.show()

    # === STEP: Get IBIs from peaks
    ibis = sp.get_ibis(peaks)
    
    if with_plots:
        plt.title('IBIs')
        plt.plot(ibis)
        plt.show()

    # === STEP: Get HR
    mibis = np.mean(ibis)
    print('Pipeline Mean IBI:', mibis)
    print('Pipeline HR:', sp.get_hr(ibis))
    print()


if __name__ == '__main__':
    pipeline(path = './channel_data/ieee-subject_001-trial_001-channel_data.csv')
