import numpy as np
import pandas as pd
import pywt
import scipy
from tqdm import tqdm
from extract_features import get_features_from_signals_peaks
from ml import Unsupervised
from wavelet_util import *

from chrominance import get_signal_and_peaks, my_get_peaks
import signal_pross as sp

GOLDEN_RATIO = 8  # ratio of BVP measurement rate --> frame rate --> hr measurement rate


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def test(subj, trial, wave, plot = False, verbose = True,
                signal_start = 2000):

        bvp_fname = f'validation_data/IEEE_data/subject_{subj}/trial_{trial}/empatica_e4/BVP.csv'
        hr_fname = f'validation_data/IEEE_data/subject_{subj}/trial_{trial}/empatica_e4/HR.csv'
            

        signal, peaks = good_get_signal_peaks(subj, trial)
        orig_signal = signal.copy()
        
        
        # get starting indices and ending indices for everything
        signal_end = signal_start + 600
        hr_start = signal_start // GOLDEN_RATIO
        hr_end = signal_end // GOLDEN_RATIO
        
        signal = signal[signal_start: signal_end]  # 20 seconds of signal
        peaks = [p - signal_start for p in peaks if signal_start <= p < signal_end]

        if verbose:
            print('HR from raw: ', get_hr(peaks))
        # plt.plot(signal)
        # plt.scatter(peaks, [signal[p] for p in peaks], marker='x', c='r')
        # plt.show()
        
        den = wavelet_denoise(signal, wavelet=wave, level = 2)[0]
        cA_peaks = get_wavelet_peaks(den)
        # targ = 70
        # perc = np.percentile(den, targ)
        # print(f'{targ}th percentile: {perc}; num peaks exceeding: {len([p for p in cA_peaks if den[p] > perc])} out of {len(cA_peaks)})')
        # cA_peaks = [p for p in cA_peaks if den[p] > perc]
        
        feats = get_features_from_signals_peaks(den, cA_peaks)
        def _get_model():
            import pickle
            with open('mlp.sav', 'rb') as fo:
                model = pickle.load(fo)
            fo.close()
            return model
        model = _get_model()
        preds = model.predict(feats)
        cA_peaks = [cA_peaks[i] for i in range(len(cA_peaks)) if preds[i] > 0.5]

        if verbose:
            print('HR from cA: ', get_hr(cA_peaks, factor = len(signal) / len(den)))
        if plot:
            plt.plot(den)
            plt.scatter(cA_peaks, [den[p] for p in cA_peaks], marker='x', c='r')
        
        

        hr = pd.read_csv(hr_fname)
        hr = hr[list(hr.columns)[0]][1: ]
        hr = hr[hr_start: hr_end]
        
        if verbose:
            print('Ground Truth HR:', np.mean(hr))

        err = abs(np.mean(hr) - get_hr(cA_peaks, factor = len(signal) / len(den)))
        
        if verbose:
            print('=== ERROR: ', err)
            print()
        
        if plot:
            plt.show()

        return err
    
    errs = []
    for subj in ['001', '002', '003', '004', '005', '006', '007']:
        for trial in ['001']:
            err = test(subj, trial, 'db10', signal_start = 2000, plot = True)
            errs.append(err)
    
    print('Average Error: ', np.mean(errs))

