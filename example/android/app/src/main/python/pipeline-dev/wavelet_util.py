from chrominance import get_signal_and_peaks, my_get_peaks
import signal_pross as sp
import numpy as np
import pywt
import scipy
    
def wavelet_denoise(signal, wavelet='db1', level=6, mode='per', threshold_type='soft', threshold_scale=1.0):
    
    vs = []

    sig = signal.copy()
    for _ in range(level):
        sig, cD = pywt.dwt(sig, wavelet)
        vs.append((sig, pywt.idwt(sig, None, wavelet)))

    return vs[-1]

def get_hr(peaks, factor = None):

    ibis = sp.get_ibis(peaks)

    if factor is not None:
        ibis = [i * factor for i in ibis]

    hr = sp.get_hr(ibis)
    return hr


def get_wavelet_peaks(sig):

    peaks = scipy.signal.find_peaks(sig)[0]
    return np.array([p for p in peaks if sig[p] > 0])

# interpolate for all nan values
def interpolate(sig):
    for i in range(len(sig)):
        if np.isnan(sig[i]):
            
            if i == 0:
                j = i + 1
                while np.isnan(sig[j]): j += 1
                sig[i] = sig[j]
            
            elif i == len(sig) - 1:
                j = i - 1
                while np.isnan(sig[j]): j -= 1
                sig[i] = sig[j]

            else:
                lower = sig[i - 1]
                j = i + 1
                while np.isnan(sig[j]): j += 1
                upper = sig[j]
                sig[i] = np.mean([lower, upper])
    return sig

def good_get_signal_peaks(subj, trial, split_num = None):
    fname = f'channel_data3/ieee-subject-{subj}-trial-{trial}.csv'
    signal, peaks = get_signal_and_peaks(fname, split_num = split_num)

    signal = interpolate(signal)
    peaks = my_get_peaks(signal)
    return signal, peaks