import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieee_video_start_frames import StartFrames

RGB_RATE = 30
BVP_RATE = 64


class IeeeGroundTruth:

    def __init__(self, subject, trial):
        self.subject = subject
        self.trial = trial
        self.directory = 'channel_data'
    
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

        # get the event markers
        event_markers = pd.read_csv(
            f'validation_data/IEEE_data/subject_{self.subject:03d}/trial_{self.trial:03d}/empatica_e4/tags.csv'
        ).to_numpy()

        start_gap =  event_markers[0] - bvp[0] 
        end_point =  event_markers[1] - event_markers[0]

        start_frame = StartFrames[self.subject][self.trial]

        rgb_align = rgb[start_frame: start_frame + int(end_point * 30) + 5]
        bvp_align = bvp[int(start_gap * 64): int((start_gap + end_point) * 64)]

        return rgb_align, bvp_align


    @staticmethod
    def bvp_to_ibi(self, bvp):
        pass


if __name__ == '__main__':
    for S in StartFrames:
        a, b = IeeeGroundTruth(S, 1).align_rgb_bvp()
        plt.plot(b[5000: 7000])
        plt.show()
        # print(S, 1)
        # print(a.shape[0] / RGB_RATE)
        # print(b.shape[0] / BVP_RATE)
        # print()

