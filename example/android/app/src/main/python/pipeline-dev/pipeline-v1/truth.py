import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, resample

from ieee_video_start_frames import StartFrames

RGB_RATE = 30
BVP_RATE = 64


class IeeeGroundTruth:

    def __init__(self, subject, trial, directory = 'channel_data'):
        self.subject = subject
        self.trial = trial
        
        self.directory = directory
        self.rgb_freq = RGB_RATE
        self.bvp_freq = BVP_RATE
    
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
        # downsampled_fr = float(self.bvp_freq / self.downsample_factor)
        # downsampled_to_fgb_freq_ratio = downsampled_fr / self.rgb_freq
        # return int(rgb_index * downsampled_to_fgb_freq_ratio)
        return int(rgb_index * (self.bvp_freq / self.rgb_freq))
