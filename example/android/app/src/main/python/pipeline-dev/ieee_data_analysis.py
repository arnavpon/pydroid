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

class IeeeAnalyzer:
    
    def __init__(self, subject, trial):

        self.subject = subject
        self.trial = trial

        self.video_path = f'{VAL_DATA_PATH}/{subject}/{trial}/video/video.MOV'
        self.channel_path = f'{CHANNEL_DATA_PATH}/ieee-{subject}-{trial}-channel_data.csv'
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
        return df.iloc[2: ].to_numpy()

    def get_channels(self, show_frames = True, S = None):
        track_video(self.video_path, self.channel_path, show_frames = show_frames, S = S)


    def ieee_pipeline(self, lim = 0):
        pipeline(self.channel_path, lim = lim)
    



if __name__ == '__main__':
    
    sub = 'subject_005'
    trial = 'trial_001'
    
    ieee = IeeeAnalyzer(sub, trial)

    print('Ground truth mean HR from HR vector:', np.mean(ieee.get_hr()))
    print('Ground truth HR from IBIs', 60 / np.mean(ieee.get_ibis()))
    
    ieee.get_channels(show_frames = True, S = 120)
    ieee.ieee_pipeline(lim = 2000)
