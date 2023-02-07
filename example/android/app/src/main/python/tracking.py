import cv2
import dlib
import numpy as np
import pickle
import pandas as pd
import os
from tqdm import tqdm

from forehead import find_forehead
from FaceDetection_MT1 import (
    FacialImageProcessing,
    TAKEN_W, TAKEN_H,
    LOADED_W, LOADED_H,
    IMG_THRESH
)

DEFAULT_ROI_PERCENTAGE = 0.5
DEFAULT_CSV_NAME = './channels.csv'
AUTOGEN_CCV = './channels2.csv'


def track_video(video_path = None):
    """
    Based on: https://xailient.com/blog/learn-how-to-create-a-simple-face-tracking-system-in-python/
    """

    # load video with cv2
    if video_path is None:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(video_path)
    
    resize_img = video_path is not None

    bbox, success, frame = _get_forehead_bbox(video, resize = resize_img)
    if bbox is None:
        print('[Python] Face not found in first frame')
        return
    
    # init the tracker
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2']))
    tracker.start_track(frame, rect)
    
    # display the first frame
    _display_frame(frame, bbox)

    # init dict to track spatial average of face ROI for each color channel
    channel_data = {'r': [], 'g': [], 'b': []}

    while success:
        
        # read the next frame
        success, frame = video.read()

        if success:

            # get new bbox for the current frame
            curr_img = _track(frame, tracker, channel_data, resize_img)

            # display current frame
            _display_frame(curr_img)

            if cv2.waitKey(10) == 27:
                break
    
    pd.DataFrame(data = channel_data).to_csv(DEFAULT_CSV_NAME, index = False)
    video.release()


def _track(frame, tracker, channel_data, resize_img):

    img = _process_img(frame, resize = resize_img)

    # make a copy of the frame
    img = np.copy(frame)
    
    # update the tracker based on the current image
    tracker.update(img)
    pos = tracker.get_position()

    # isolate starting and ending x/y
    start_x = int(pos.left())
    start_y = int(pos.top())
    end_x = int(pos.right())
    end_y = int(pos.bottom())

    bbox = {'x1': start_x, 'y1': start_y, 'x2': end_x, 'y2': end_y}
    _collect_channel_data(channel_data, img, bbox)

    #start_x, end_x, start_y, end_y = _get_cropped_channels(img, {'x1': start_x, 'y1': start_y, 'x2': end_x, 'y2': end_y}, DEFAULT_ROI_PERCENTAGE)

    # draw new rectangle on the img
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
    
    return img

def _collect_channel_data(channel_data, img, bbox, roi_percentage = DEFAULT_ROI_PERCENTAGE):
    """
    Populate the given dictionary by appending a spatial average for each channel
    within a percentage of the ROI found for the face. The reason for collecting a percentage
    as sometimes the outer portions of the ROI contain hair, eyebrows, etc.
    """

    # get channels from image which is cropped by the bbox produced which is shrunk by 1 - roi_percentage
    rc, gc, bc = _get_cropped_channels(img, bbox, roi_percentage)

    # for each channel, append its mean to the channel dict
    channel_data['r'].append(np.mean(rc))
    channel_data['g'].append(np.mean(gc))
    channel_data['b'].append(np.mean(bc))

def _get_cropped_channels(img, bbox, roi_percentage):

    roi_width = bbox['x2'] - bbox['x1']
    roi_height = bbox['y2'] - bbox['y1']

    xi = int(bbox['x1'] + ((roi_width * (1 - roi_percentage)) / 2))
    xw = int(roi_width * roi_percentage)
    # xii = xi + xw
    yi = int(bbox['y1'] + ((roi_height * (1 - roi_percentage)) / 2))
    yh = int(roi_height * roi_percentage)
    # yii = yi + yw

    # crop the image and split by channel
    cropped_img = img[yi: yi + yh, xi: xi + xw]
    return cv2.split(cropped_img)

def _get_forehead_bbox(vid, W = 20, resize = False):

    # search the first W frames for a face
    for i in tqdm(range(W)):
        
        success, frame = vid.read()
        if not success:
            print("Video ended before face was found!")
            return None

        img = _process_img(frame, resize = resize)

        # load facial image recognition class and get initial bounding box
        imgProcessing = FacialImageProcessing(False)
        bounding_boxes, _ = imgProcessing.detect_faces(img)

        if len(bounding_boxes) > 0:
            bbox = bounding_boxes[0]
            bbox_dict = {'x1': bbox[0],'y1': bbox[1],'x2': bbox[2],'y2': bbox[3]}
            print("Bounding Boxes: {0}".format(bbox_dict))
        else:
            print("No face detected!")
            return None

        # adjust y2 value so that box encloses just the user's forehead
        try:
            bbox_dict['y2'] -= find_forehead(img, bbox_dict)
            return bbox_dict, success, frame
        except: continue

    return None


def _process_img(frame, resize = False):

    # convert the image to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # resize image differently depending on if it's taken locally (<IMG_THRESH) or loaded from the library
    if resize:
        img = (
            cv2.resize(img, (int(TAKEN_W), int(TAKEN_H)), interpolation = cv2.INTER_AREA)
            if img.shape[0] < IMG_THRESH
            else cv2.resize(img, (int(LOADED_W), int(LOADED_H)), interpolation = cv2.INTER_AREA)
        )

    return img

def _display_frame(img, bbox = None):
    if bbox is not None:
        cv2.rectangle(
            img,
            (int(bbox['x1']), int(bbox['y1'])),
            (int(bbox['x2']), int(bbox['y2'])),
            (0, 255, 0),
            3
        )
    
    cv2.imshow('Image', img)


if __name__ == '__main__':
    # track_video('/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/Movie on 2-2-23 at 3.31 PM.mp4')
    track_video()
    # track_video('/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/PXL_20230202_212010481.mp4')
