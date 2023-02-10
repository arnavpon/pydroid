import cv2
import numpy as np
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from GetHR import (
    pipeline,
    detrend_w_poly,
    normalize_detrended,
    n_moving_avg,
    get_peaks_simple,
    get_ibis,
    get_hr
)
from tracking import track_video, display_frame, process_img, collect_channel_data
from FaceDetection_MT1 import FacialImageProcessing

VAL_DATA_PATH = '/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/validation_data/IEEE_data'
CHANNEL_DATA_PATH = '/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/channel_data'

def get_subject_ibis(subject = 'subject_001', trial = 'trial_001'):
    
    path = f'{VAL_DATA_PATH}/{subject}/{trial}/empatica_e4/IBI.csv'
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    
    ibi_vect = df['IBI'].to_numpy()
    print('Ground truth mean IBIs', np.mean(ibi_vect))
    print('Ground truth subject HR:', 60 / np.mean(ibi_vect))


def ieee_channels(subject = 'subject_001', trial = 'trial_001', S = 120):
    """
    Get channels from .csv file.
    """
    
    path = f'{VAL_DATA_PATH}/{subject}/{trial}/video/video.MOV'
    channel_filepath = f'{CHANNEL_DATA_PATH}/ieee-{subject}-{trial}-channel_data.csv'

    track_video(path, channel_filepath, show_frames = True, S = S)


def ieeeline(path):
    pipeline(path, lim = 500)
    



if __name__ == '__main__':
    sub = 'subject_005'
    trial = 'trial_001'
    ieee_channels(subject = sub, trial = trial, S = 120)
    get_subject_ibis(subject = sub, trial = trial)
    ieeeline(f'/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/channel_data/ieee-{sub}-{trial}-channel_data.csv')