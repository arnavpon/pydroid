"""
Module handling the calculation of HR and HRV using color channel vectors
derived from a face video.
"""

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.sparse import diags

from jadeR import jadeR
from ica import vanilla_ica, jade
from tracking import DEFAULT_CSV_NAME


def load_channels(path = DEFAULT_CSV_NAME):
    """
    Step 1: Load ROI channel vectors from local csv file.
    """
    
    channels = pd.read_csv(path)
    if (
        'r' not in channels.columns
        or 'g' not in channels.columns
        or 'b' not in channels.columns
    ):
        raise Exception('Channels data must contain r, g, and b data.')

    return {
        'r': channels.r.to_numpy(),
        'g': channels.g.to_numpy(),
        'b': channels.b.to_numpy()
    }


def detrend(channel, smoothing_param = 10):
    """
    Step 2: Given a color channel vector, apply detrending as described in 
    "An Advanced Detrending Method With Application to HRV Analysis"
    """

    # make sure the channel is a np array
    if not isinstance(channel, np.array):
        channel = np.array(channel)
    
    # get length of the vector
    T = len(channel)

    I = np.eye(T)
    D2 = diags(
        [1, -2, 1],
        [0, 1, 2],
        shape = (T - 2, T)
    ).toarray()

    z_stat = np.dot(
        np.eye(T) - np.linalg.inv(I + (smoothing_param ** 2) * np.dot(D2.T, D2)),
        channel
    )

    return z_stat

    
def normalize_detrended(detrended_channel):
    """
    Step 3: Input is detrended channel from step 2. This method implements
    formula 3 in "Advancements in Noncontact, Multiparameter Physiological
    Measurements Using a Webcam".
    """

    # make sure the channel is a np array
    if not isinstance(detrended_channel, np.array):
        detrended_channel = np.array(detrended_channel)

    # get mean and standard dev of the channel
    mn = np.mean(detrended_channel)
    std = np.std(detrended_channel)

    return (detrended_channel - mn) / std


def get_bvp_w_ica(X, ica_method = vanilla_ica):
    """
    
    """

    # decompose normalized raw traces
    components = ica_method(X)
    
    # collect power spectra from each component
    power_spectra = []
    for comp in components:
        _, Pxx = welch(comp, fs = 1000)
        power_spectra.append(Pxx)

    # return the component w the largest peak power spectra
    bvp_index = np.argmax([np.max(ps) for ps in power_spectra])
    return components[bvp_index]

if __name__ == '__main__':

    channels = load_channels()
    print(jade(channels))
