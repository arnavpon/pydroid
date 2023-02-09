import cv2
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from GetHR import (
    detrend_w_poly,
    normalize_detrended,
    n_moving_avg,
    get_peaks_simple,
    get_ibis,
    get_hr
)
from tracking import track_img, display_frame, process_img, collect_channel_data
from FaceDetection_MT1 import FacialImageProcessing

EXAMPLE_PATH = '/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/px1_full (1).mat'


def get_pulseox_array(path):
    """
    Get pulse ox array from .mat file.
    """
    mat = io.loadmat(path)
    return mat['pulseOxRecord'][0]

def analyze_frames(base_dir = './validation_data/rice_data', subject = 'subj_010', show_frames = False, roi_percentage = 0.5):
    directory = f'{base_dir}/{subject}'

    files = [f'{directory}/{f}' for f in os.listdir(directory) if f[0] != '.']
    files.sort()
    
    channel_data = {
        'r': [],
        'g': [],
        'b': []
    }
    for i in tqdm(range(len(files))):
        f = files[i]

        res = track_img(f)

        if res is None:
            continue
        else:
            img, bbox = res
            smaller_bbox = collect_channel_data(channel_data, img, bbox, roi_percentage = roi_percentage)

            if show_frames:
                cv2.rectangle(img, (smaller_bbox['x1'], smaller_bbox['y1']), (smaller_bbox['x2'], smaller_bbox['y2']), (0, 0, 255), 2)
                cv2.rectangle(img, (int(bbox['x1']), int(bbox['y1'])), (int(bbox['x2']), int(bbox['y2'])), (0, 255, 0), 2)
                cv2.imshow('img', img)
                cv2.waitKey(1)

    pd.DataFrame(channel_data).to_csv(
        f'/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/channel_data/rice-{subject}-channel_data.csv',
        index = False
    )


def riceline(path = EXAMPLE_PATH, preproesess = True):

    signal = get_pulseox_array(path)
    print('signal len:', len(signal))
    
    if preproesess:
        signal = detrend_w_poly(signal)
        signal = normalize_detrended(signal)
        signal = n_moving_avg(signal, window = 15)
    

    peaks = get_peaks_simple(signal)
    ibis = get_ibis(peaks, fr = 60)
    hr = get_hr(ibis)
    print('HR:', hr)

    plt.plot(signal[0: 1000])
    p = [i for i in peaks if i < 1000]
    print(p)
    plt.scatter(p, [signal[i] for i in p], marker = 'x', color = 'red')
    plt.show()


if __name__ == '__main__':
    analyze_frames(base_dir = './validation_data/rice_data', subject = 'subj_015', show_frames = False)
