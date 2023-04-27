import cv2
import dlib
import numpy as np
from scipy.signal import welch

import sys
sys.path.insert(0, './SkinDetector')
import skin_detector
import signal_pross as sp

from FaceDetection_MT1 import (
    FacialImageProcessing,
    TAKEN_W, TAKEN_H,
    LOADED_W, LOADED_H,
    IMG_THRESH
)
import matplotlib.pyplot as plt

DEFAULT_PATH = './validation_data/IEEE_data/subject_002/trial_001/video/video.MOV'
FRAME_WINDOW = (200, 600)


def get_hr(vecs, framerate = 30, nsegments = 12, range_min = 0.9, range_max = 1.8):

    l = int(framerate * 1.6)
    H = np.zeros(vecs.shape[0])

    for t in range(0, (vecs.shape[0] - l)):

        # === Step 1: Spatial averaging
        C = vecs[t: t + l - 1, :].T
        
        # === Step 2 : Temporal normalization
        mean_colors = np.mean(C, axis = 1) 
        diag_mean_colors = np.diag(mean_colors)
        diag_mean_colors_inv = np.linalg.inv(diag_mean_colors)
        Cn = np.matmul(diag_mean_colors_inv, C)
    
        #Step 3: Projection matrix
        projection_matrix = np.array([[0,1,-1], [-2,1,1]])
        S = np.matmul(projection_matrix,Cn)

        # === Step 4:
        std = np.array([1, np.std(S[0,:]) / np.std(S[1,:])])
        P = np.matmul(std, S)

        # === Step 5: Overlap-Adding
        H[t: t + l - 1] = H[t: t + l - 1] +  (P - np.mean(P)) / np.std(P)

    signal = H
    segment_length = (2 * signal.shape[0]) // (nsegments + 1) 
    green_f, green_psd = welch(signal, framerate, 'flattop', nperseg = segment_length) #, scaling='spectrum',nfft=2048)
    # NOTE: Use power spectral density calculation to filter out high freq noise
    # NOTE: revisit fast fourier transform
    
    first = np.where(green_f > range_min)[0] #0.8 for 300 frames
    last = np.where(green_f < range_max)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)

    max_idx = np.argmax(green_psd[range_of_interest])
    f_max = green_f[range_of_interest[max_idx]]

    hr = f_max * 60.0
    return hr

def preprocess(v, window = 5):
    detrendeds = []
    normeds = []
        
    for j in range(v.shape[1]):
        new_v = v[:, j]
        print(new_v.shape)
        print(new_v)
        detrended = sp.detrend_w_poly(new_v)
        normed = sp.normalize_detrended(detrended)

        detrendeds.append(detrended)
        normeds.append(normed)

        v[:, j] = normed
    
    for d in detrendeds:
        plt.plot(d)
    plt.show()
    for n in normeds:
        plt.plot(n)
    plt.show()

    return v

    
def acquire_raw_signals(path, framerate = 30, with_tracker = True, color_filter = cv2.COLOR_BGR2RGB):

    vid = cv2.VideoCapture(path)

    frame_num = 0
    success, img, bbox = _get_bbox(vid, frame_num, tracker = None)

    if not success:
        print('Could not read HR.')
        return
    
    if with_tracker:
        tracker = _init_tracker(img, bbox)
    else: 
        tracker = None

    vecs = None
    while success:
        if frame_num % 2000 == 0: print(f'On frame: {frame_num}')# and frame_num < FRAME_WINDOW[1]:
        
        # if frame_num < FRAME_WINDOW[0]:
        #     success, img, bbox = _get_bbox(vid, tracker = tracker)
        #     frame_num += 1
        #     continue

        masked_face, mask = _crop_and_mask(img, bbox)
        val1, val2, val3 = _get_frame_vector(masked_face, mask)  # typically r, g, and b
        
        if vecs is None:
            vecs = np.array([val1, val2, val3])
        else:
            vecs = np.vstack((vecs, np.array([val1, val2, val3])))

        success, img, bbox = _get_bbox(vid, frame_num, tracker = tracker)
        frame_num += 1
    
    vid.release()            
    return vecs


def _get_bbox(vid, frame_num, color_filter = cv2.COLOR_BGR2RGB, iter_lim = 100, tracker = None):

    success, frame = vid.read()
    if not success:
        return False, None, None
    img = cv2.cvtColor(frame, color_filter)

    if tracker is None:
        return _get_bbox_w_model(vid, img, color_filter, iter_lim, frame_num)
    
    else:
        
        tracker.update(img)
        pos = tracker.get_position()

        start_x = int(pos.left())
        start_y = int(pos.top())
        end_x = int(pos.right())
        end_y = int(pos.bottom())

        return success, img, {'x1': start_x, 'y1': start_y, 'x2': end_x, 'y2': end_y}


def _get_bbox_w_model(vid, img, color_filter, iter_lim, frame_num):

    # search the first iter_lim frames for a face
    bbox_dict = None
    success = True
    for _ in range(iter_lim):

        imgProcessing = FacialImageProcessing(False)
        bounding_boxes, _ = imgProcessing.detect_faces(img)
        
        if len(bounding_boxes) == 0:
            success, frame = vid.read()
            frame_num += 1
            if success:
                img = cv2.cvtColor(frame, color_filter)
            else:
                return success, None, None
        
        else:

            bbox = bounding_boxes[0]
            return success, img, {'x1': bbox[0],'y1': bbox[1],'x2': bbox[2],'y2': bbox[3]}

def _init_tracker(frame, bbox):
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2']))
    tracker.start_track(frame, rect)
    return tracker

def _crop_and_mask(img, bbox, roi_percentage = 1):

    roi_width = bbox['x2'] - bbox['x1']
    roi_height = bbox['y2'] - bbox['y1']

    xi = max(0, int(bbox['x1'] + ((roi_width * (1 - roi_percentage)) / 2)))
    xw = int(roi_width * roi_percentage)
    yi = max(0, int(bbox['y1'] + ((roi_height * (1 - roi_percentage)) / 2)))
    yh = int(roi_height * roi_percentage)
    
    face = img[yi: yi + yh, xi: xi + xw]
    mask = skin_detector.process(face)
    masked_face = cv2.bitwise_and(face, face, mask = mask)

    return masked_face, mask

def _get_frame_vector(masked_face, mask):
    """
    Given the face and mask, calculate the mean for each channel.
    """

    number_of_skin_pixels = np.sum(mask > 0)
    r = np.sum(masked_face[:,:,2]) / number_of_skin_pixels
    g = np.sum(masked_face[:,:,1]) / number_of_skin_pixels 
    b = np.sum(masked_face[:,:,0]) / number_of_skin_pixels
    return (r, g, b)


if __name__ == '__main__':
    v = acquire_raw_signals(DEFAULT_PATH)
    print(v.shape)
    plt.plot(v)
    plt.show()
    # v = preprocess(v)
    # print(v.shape)
    # plt.plot(v)
    # plt.show()
    # hr = get_hr(v)
    # print(hr)
