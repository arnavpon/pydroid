import cv2
import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from pipeline import pipeline
from tracking import track_video, display_frame, process_img, collect_channel_data
from FaceDetection_MT1 import FacialImageProcessing

VAL_DATA_PATH = '/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/pipeline-dev/validation_data/IEEE_data'
CHANNEL_DATA_PATH = '/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/pipeline-dev/channel_data'
YUV_CHANNEL_DATA_PATH = '/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/pipeline-dev/channel_data_yuv'

class IeeeAnalyzer:
    
    def __init__(self, subject, trial, channel_data_path = None):

        self.subject = subject
        self.trial = trial

        self.video_path = f'{VAL_DATA_PATH}/{subject}/{trial}/video/video.MOV'
        self.channel_path = f'{CHANNEL_DATA_PATH if channel_data_path is None else channel_data_path}/ieee-{subject}-{trial}-channel_data.csv'
        self.csv_data_path = f'{VAL_DATA_PATH}/{subject}/{trial}/empatica_e4/'
    
    def get_ibis(self):
        """
        Get the inter-beat intervals from the ground truth data.
        """
        
        path = f'{self.csv_data_path}/IBI.csv'
        df = pd.read_csv(path)
        df.columns = [col.strip() for col in df.columns]
        ibi_vect = df['IBI'].to_numpy()
        return ibi_vect

    def get_hr(self):
        """
        Get the heart rate from the ground truth data.
        """
        
        path = f'{self.csv_data_path}/HR.csv'
        df = pd.read_csv(path)
        df.columns = [col.strip() for col in df.columns]
        return np.mean(df.iloc[2: ].to_numpy())

    def get_channels(self, show_frames = True, S = None):
        track_video(self.video_path, self.channel_path, show_frames = show_frames, S = S)


    def ieee_pipeline(self, moving_average_window = 5, ncvt_tolerance = 0.03, ncvt_step_size = 50, lim = 0, with_plots = True):
        return pipeline(
            self.channel_path,
            moving_average_window = moving_average_window,
            ncvt_tolerance = ncvt_tolerance,
            ncvt_step_size = ncvt_step_size,
            lim = lim,
            with_plots = with_plots
        )
    

def validate(sub_trials,
             ground_truth_method = 'HR',
             moving_average_window = 5,
             ncvt_tolerance = 0.03,
             ncvt_step_size = 50,
             lim = 500,
             with_plots = False):
    """
    Validate the pipeline on the IEEE data set.
    """

    if ground_truth_method not in ['HR', 'IBI']:
        raise ValueError('ground_truth_method must be one of [HR, IBI]')
    
    errs = []
    for sub in sub_trials:
        for trial in sub_trials[sub]:
            
            ieee = IeeeAnalyzer(sub, trial)
            hr = ieee.ieee_pipeline(
                moving_average_window = moving_average_window,
                ncvt_tolerance = ncvt_tolerance,
                ncvt_step_size = ncvt_step_size,
                lim = lim,
                with_plots = with_plots
            )

            if ground_truth_method == 'HR':
                ground_truth = ieee.get_hr()
            elif ground_truth_method == 'IBI':
                ground_truth = 60 / ieee.get_ibis()
            
            err = np.abs(hr - ground_truth)
            print(f'Error for {sub}-{trial}: {err}\n\n')
            errs.append(err)
    
    print('MSE: ', np.mean([pow(err, 1) for err in errs]))
    



if __name__ == '__main__':

    # Sub 3, trial 1: 78.26086956521739

    sub_trials = {
        'subject_001': ['trial_001'],
        'subject_002': ['trial_001', 'trial_002', 'trial_003'],
        'subject_003': ['trial_001', 'trial_002'],
        'subject_004': ['trial_001'],
        'subject_005': ['trial_001'],
        'subject_006': ['trial_001'],
        'subject_007': ['trial_001'],
    }
    
    for sub in sub_trials:
        for trial in sub_trials[sub]:
            print(f'On {sub}-{trial}')
            ieee = IeeeAnalyzer(sub, trial, channel_data_path = YUV_CHANNEL_DATA_PATH)
            print('Ground truth mean HR from HR vector:', np.mean(ieee.get_hr()))
            print('Ground truth HR from IBIs', 60 / np.mean(ieee.get_ibis()))
            print()
    #         ieee.get_channels(show_frames = True, S = 150)

            # tol = 0.03
            # hr = ieee.ieee_pipeline(
            #     moving_average_window = 5,
            #     ncvt_tolerance = tol,
            #     ncvt_step_size= 10,
            #     lim = 800,
            #     with_plots = True
            # )
            # print(f'Pipeline HR: {hr}')
            # print()
