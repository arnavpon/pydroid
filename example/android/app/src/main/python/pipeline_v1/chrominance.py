"""
Implementation of the chrominance method for extracting a raw rPPG signal.
Taken from "Robust Pulse Rate From Chrominance-Based rPPG" by de Haan and Jeanne.

Refactored March 6, 2023
"""

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple

# local modules
from pipeline_v1.signal_pross import bandpass, detrend_w_poly, normalize_signal


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


def chrominance(sig: str or np.array, settings: dict = CHROM_SETTINGS, 
                bounds: Tuple[int, int] = (0, -1)):
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
    
    # apply generic detrending and normalization to the raw signals
    r = detrend_w_poly(r)
    g = detrend_w_poly(g)
    b = detrend_w_poly(b)
    r = normalize_signal(r)
    g = normalize_signal(g)
    b = normalize_signal(b)

    # normalize skin tones
    def _tonenorm(v):
        return v / np.sqrt(pow(r, 2) + pow(g, 2) + pow(b, 2))
    r_n, g_n, b_n = _tonenorm(r), _tonenorm(g), _tonenorm(b)

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
