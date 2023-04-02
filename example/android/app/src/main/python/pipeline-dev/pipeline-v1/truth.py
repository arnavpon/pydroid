import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, resample

from chrominance import chrominance, CHROM_SETTINGS
from ieee_video_start_frames import StartFrames
from signal_pross import (
    normalize_signal,
    detrend_w_poly,
    normalize_amplitude_to_1,
    bandpass,
    min_max_scale
)

from wavelet import apply_wavelet

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
        
        def _tonenorm(rgb, idx):
            return rgb[:, idx] / (np.sqrt(
                pow(rgb[:, 0], 2) + pow(rgb[:, 1], 2) + pow(rgb[:, 2], 2)
            ))

        for i in range(self.rgb.shape[1]):

            # normalize, detrend, and then set amplitude to 1
            self.rgb[:, i] = detrend_w_poly(self.rgb[:, i])
            # self.rgb[:, i] = min_max_scale(self.rgb[:, i])
            # self.rgb[:, i] = normalize_signal(self.rgb[:, i])
            # self.rgb[:, i] = normalize_amplitude_to_1(self.rgb[:, i])
        
        # for i in range(self.rgb.shape[1]):
        #     self.rgb[:, i] = _tonenorm(self.rgb, i)
        
        for i in range(self.rgb.shape[1]):
            self.rgb[:, i] = apply_wavelet(self.rgb[:, i], cutoff_low = 0.5, cutoff_high = 3, wave = 'db2', level = 1)
        
        for i in range(self.rgb.shape[1]):
            self.rgb[:, i] = bandpass(self.rgb[:, i], self.rgb_freq, [0.5, 3], order = 4)

        for i in range(self.rgb.shape[1]):
            self.rgb[:, i] = min_max_scale(self.rgb[:, i])

    
    def process_bvp(self):
            
        # normalize, detrend, and then set amplitude to 1
        self.bvp = normalize_signal(self.bvp)
        self.bvp = detrend_w_poly(self.bvp)
        # self.bvp = min_max_scale(self.bvp)
        self.bvp = normalize_amplitude_to_1(self.bvp)
        
    # def interpolate_bvp(self):
    #     self.bvp_interp = interp1d(
    #         np.arange(len(self.bvp)) * self.bvp_freq / self.rgb_freq, self.bvp
    #     )(np.arange(self.rgb.shape[0]))
    
    # def prepare_diffs(self):

    #     if self.rgb is None or self.bvp_interp is None:
    #         raise Exception('Must call align_rgb_bvp and interpolate_bvp before calling prepare_diffs.')

    #     self.rgb_diffs = self.rgb[1: , :].copy()
    #     for i in range(self.rgb_diffs.shape[1]):
    #         self.rgb_diffs[:, i] = np.diff(self.rgb[:, i])
        
    #     self.rgb = self.rgb[1: , :]
    #     self.bvp_interp = self.bvp_interp[1: ]
    
    def prepare_data_for_ml(self):
        """
        Put the rgb and bvp data together into a single dataframe
        """

        # get upsampled rgb
        rgb_upsampled = np.zeros((len(self.bvp), self.rgb.shape[1]))
        for col in range(rgb_upsampled.shape[1]):
            rgb_upsampled[:, col] = self.upsample_signal(
                self.rgb[:, col],
                len(self.bvp),
                old_fs = self.rgb_freq,
                new_fs = self.bvp_freq
            )

        # get "velocity" and "acceleration" of rgb
        rgb_vel = np.zeros((rgb_upsampled.shape[0] - 2, rgb_upsampled.shape[1]))
        rgb_acc = np.zeros((rgb_upsampled.shape[0] - 2, rgb_upsampled.shape[1]))
        for col in range(rgb_vel.shape[1]):

            vel = np.diff(rgb_upsampled[:, col], n = 1)[1: ]
            acc = np.diff(rgb_upsampled[:, col], n = 2)
            rgb_vel[:, col] = vel
            rgb_acc[:, col] = acc


        # exclude first element of both upsampled rgb and corresponding bvp
        rgb_upsampled = rgb_upsampled[2: , :]
        chrom = self.prepare_chrominance_as_feature(rgb_upsampled)
        bvp_in_use = self.bvp[2: ]

        data = {
            'chrom': chrom,
            'r': rgb_upsampled[:, 0],
            'g': rgb_upsampled[:, 1],
            'b': rgb_upsampled[:, 2],
            'r_vel': rgb_vel[:, 0],
            'g_vel': rgb_vel[:, 1],
            'b_vel': rgb_vel[:, 2],
            'r_acc': rgb_acc[:, 0],
            'g_acc': rgb_acc[:, 1],
            'b_acc': rgb_acc[:, 2],
            'bvp': bvp_in_use
            # 'bvp': np.diff(bvp_in_use)
        }

        # # make columns all the same length
        # for col in data:
        #     if col != 'bvp':
        #         data[col] = data[col][1: ]

        return pd.DataFrame(data)
    
    def preprocess_original(self):

        data = self.prepare_data_for_ml()

        for col in data.columns:
            data[col] = normalize_signal(data[col].to_numpy())
            data[col] = detrend_w_poly(data[col].to_numpy())
            data[col] = normalize_amplitude_to_1(data[col].to_numpy())

        return data

    def preprocess_basic(self):

        data = self.prepare_data_for_ml()

        # apply min-max scaling to each column
        for col in data.columns:
            data[col] = min_max_scale(data[col].to_numpy())
        
        return data

    def preprocess_with_wavelet(self, target_col = 'bvp', wave = 'db2', level = 1):

        data = self.prepare_data_for_ml()

        # take only the middle of the data
        divisor = 5
        interval = len(data) // divisor
        data = data[interval: -interval]

        if level > 0:
        
            # apply wavelet to each column, except targets
            for col in data.columns:
                if col != target_col:
                    data[col] = apply_wavelet(data[col].to_numpy(), wave = wave, level = level)

        # apply min-max scaling to each column
        for col in data.columns:
            data[col] = min_max_scale(data[col].to_numpy())
        
        return data



    @staticmethod
    def upsample_signal(signal, new_length, old_fs = 30, new_fs = 64):

        f = resample(signal, new_length)
        return f

    @staticmethod
    def prepare_chrominance_as_feature(rgb):

        chrom = chrominance(rgb, CHROM_SETTINGS, None, False)

        chrom = apply_wavelet(chrom, 'db2', 2)
        chrom = bandpass(chrom, 64, [0.67, 3.0], 4)
        chrom = min_max_scale(chrom)
        return chrom
