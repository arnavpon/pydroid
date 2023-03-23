import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, resample

from ieee_video_start_frames import StartFrames
from signal_pross import (
    normalize_signal,
    detrend_w_poly,
    normalize_amplitude_to_1
)

RGB_RATE = 30
BVP_RATE = 64


class IeeeGroundTruth:

    def __init__(self, subject, trial, directory = 'channel_data'):
        self.subject = subject
        self.trial = trial
        
        self.directory = directory
        self.rgb_freq = RGB_RATE
        self.bvp_freq = BVP_RATE

        self.rgb = None
        self.bvp = None

        self.bvp_interp = None
    
    def align_rgb_bvp(self):

        # get rgb
        rgb = pd.read_csv(
            f'{self.directory}/ieee-subject-{self.subject:03d}-trial-{self.trial:03d}.csv'
        ).to_numpy()
        
        # get bvp
        bvp = pd.read_csv(
            f'validation_data/IEEE_data/subject_{self.subject:03d}/trial_{self.trial:03d}/empatica_e4/BVP.csv',
            header = None
        ).to_numpy()

        # get ibi
        self.ibis = pd.read_csv(
            f'validation_data/IEEE_data/subject_{self.subject:03d}/trial_{self.trial:03d}/empatica_e4/IBI.csv',
            header = None
        ).to_numpy()[1: , 1].flatten()

        # get the event markers
        event_markers = pd.read_csv(
            f'validation_data/IEEE_data/subject_{self.subject:03d}/trial_{self.trial:03d}/empatica_e4/tags.csv'
        ).to_numpy()

        start_gap =  event_markers[0] - bvp[0] 
        end_point =  event_markers[1] - event_markers[0]

        start_frame = StartFrames[self.subject][self.trial]

        self.rgb = rgb[start_frame: start_frame + int(end_point * 30) + 5]
        self.bvp = bvp[int(start_gap * 64): int((start_gap + end_point) * 64)]
    
    def align_indices(self, rgb_index):
        return int(rgb_index * (self.bvp_freq / self.rgb_freq))

    def fill_nans(self):

        if self.rgb is None or self.bvp is None:
            raise Exception('Must call align_rgb_bvp before calling fill_nans.')

        # make sure first element of each isn't nan
        for i in range(self.rgb.shape[1]):
            if np.isnan(self.rgb[0, i]):
                for val in self.rgb[1: , i]:
                    if not np.isnan(val):
                        self.rgb[0, i] = val
                        break
        if np.isnan(self.bvp[0]):
            for val in self.bvp[1: ]:
                if not np.isnan(val):
                    self.bvp[0] = val
                    break
        
        for i in range(self.rgb.shape[1]):
            self.rgb[:, i] = pd.Series(self.rgb[:, i]).interpolate(method = 'linear').to_numpy()
        self.bvp = pd.Series(self.bvp.flatten()).interpolate(method = 'linear').to_numpy()
        
    def process_rgb(self):
        
        for i in range(self.rgb.shape[1]):

            # normalize, detrend, and then set amplitude to 1
            self.rgb[:, i] = normalize_signal(self.rgb[:, i])
            self.rgb[:, i] = detrend_w_poly(self.rgb[:, i])
            self.rgb[:, i] = normalize_amplitude_to_1(self.rgb[:, i])
    
    def process_bvp(self):
            
            # normalize, detrend, and then set amplitude to 1
            self.bvp = normalize_signal(self.bvp)
            self.bvp = detrend_w_poly(self.bvp)
            self.bvp = normalize_amplitude_to_1(self.bvp)
        
    def interpolate_bvp(self):
        self.bvp_interp = interp1d(
            np.arange(len(self.bvp)) * self.bvp_freq / self.rgb_freq, self.bvp
        )(np.arange(self.rgb.shape[0]))
    
    def prepare_diffs(self):

        if self.rgb is None or self.bvp_interp is None:
            raise Exception('Must call align_rgb_bvp and interpolate_bvp before calling prepare_diffs.')

        rgb_copy = self.rgb.copy()
        rgb_copy = rgb_copy[1: , :]
        for i in range(rgb_copy.shape[1]):
            rgb_copy[:, i] = np.diff(self.rgb[:, i])
        
        self.rgb = rgb_copy
        self.bvp_interp = self.bvp_interp[1: ]
    
    def prepare_data_for_ml(self):
        """
        Put the rgb and bvp data together into a single dataframe
        """

        data = {
            'r': self.rgb[:, 0],
            'g': self.rgb[:, 1],
            'b': self.rgb[:, 2],
            'bvp': self.bvp_interp
        }

        return pd.DataFrame(data)

