
import numpy as np
import pandas as pd
from scipy.signal import butter, resample, filtfilt, find_peaks, peak_prominences
from sklearn.decomposition import FastICA
from typing import Tuple
import xgboost as xgb
import os

from pipeline_v1.peaks import get_peaks_for_hr
from pipeline_v1.signal_pross import (
    bandpass,
    get_ibis,
    get_hr,
    get_hrv,
    min_max_scale,
    n_moving_avg,
)
from pipeline_v1.truth import IeeeGroundTruth

PRED_PEAKS_PROMINANCE = 0.28


def pipeline(rgb):
    """
    The pipeline takes RGB channel data collected from the face and returns the
    HR, HRV, raw predicted rPPG signal, and raw IBIs array.
    """

    # process the raw rgb
    rgb = rgb.to_numpy()
    rgb = IeeeGroundTruth.process_rgb_general(
        rgb,
        minmax = False,
        use_wavelet = True,
        use_bandpass = False,
        rgb_freq = 30
    )

    # perform feature engineering on the RGB data
    data = IeeeGroundTruth.prepare_data_for_ml_general(
        rgb,
        num_feats_per_channel = 8,
        skip_amount = 12
    ).to_numpy()
    print(f'Got data with shape: {data.shape}')

    # load the pre-trained model
    models_path, _ = os.path.split(os.path.realpath(__file__))
    model_file = os.path.join(models_path, 'moodboost.xgb')
    model = xgb.Booster()
    model.load_model(model_file)

    # get the prediction vector
    prediction = model.predict(xgb.DMatrix(data))

    # process the prediction vector
    orig_len = len(prediction)
    prediction = n_moving_avg(prediction, 10)
    prediction = resample(prediction, orig_len)
    prediction = bandpass(prediction, 64, [0.7, 4.0], 4)
    prediction = min_max_scale(prediction)
    
    # get the predicted peaks
    pred_peaks, _ = get_peaks_for_hr(
        prediction,
        64,
        3.0,
        -1,
        prominence = 0.28,
        with_min_dist = True,
        with_valleys = False
    )

    # get the ibis
    ibis = get_ibis(pred_peaks, fr = 64)

    # compute hr and hrv
    hr = get_hr(ibis)
    hrv = get_hrv(ibis)

    return [[hr, hrv]]
