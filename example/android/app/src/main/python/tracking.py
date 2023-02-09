import cv2
from datetime import datetime
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

def track_img(img_path):
    """
    Based on: https://xailient.com/blog/learn-how-to-create-a-simple-face-tracking-system-in-python/
    """

    img = cv2.imread(img_path)
    img = process_img(img)
    
    imgProcessing = FacialImageProcessing(False)
    bounding_boxes, _ = imgProcessing.detect_faces(img)

    if len(bounding_boxes) > 0:
        bbox = bounding_boxes[0]
        bbox_dict = {'x1': bbox[0],'y1': bbox[1],'x2': bbox[2],'y2': bbox[3]}
        # print(f'Bounding Boxes: {bbox_dict}')
    
    else:
        print('No face detected.')
        return None

    # adjust y2 value so that box encloses just the user's forehead
    try:
        bbox_dict['y2'] -= find_forehead(img, bbox_dict)
        return img, bbox_dict
    except: 
        print('No forehead detected.')
        return None

    

def track_video(video_path = None, channel_filepath = DEFAULT_CSV_NAME, show_frames = False, resize_img = False, S = 120):
    """
    Based on: https://xailient.com/blog/learn-how-to-create-a-simple-face-tracking-system-in-python/
    """

    # load video with cv2
    if video_path is None:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(video_path)

    bbox, success, frame = _get_forehead_bbox(video, resize = resize_img)
    if bbox is None:
        print('[Python] Face not found.')
        return
    
    # init the tracker
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2']))
    tracker.start_track(frame, rect)
    
    # display the first frame
    display_frame(frame, bbox)

    # init dict to track spatial average of face ROI for each color channel
    channel_data = {'r': [], 'g': [], 'b': []}

    # just loop for S seconds
    start_time = datetime.today()
    while success:
        
        if (datetime.today() - start_time).seconds > S:
            break
        
        # read the next frame
        success, frame = video.read()

        if success:

            # get new bbox for the current frame
            curr_img = _track(frame, tracker, channel_data, resize_img)

            # display current frame
            if show_frames:
                display_frame(curr_img)
                if cv2.waitKey(10) == 27:
                    break
    
    pd.DataFrame(data = channel_data).to_csv(channel_filepath, index = False)
    video.release()


def _track(frame, tracker, channel_data, resize_img):

    img = process_img(frame, resize = resize_img)

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
    condensed_bbox = collect_channel_data(channel_data, img, bbox)

    cv2.rectangle(img, (condensed_bbox['x1'], condensed_bbox['y1']), (condensed_bbox['x2'], condensed_bbox['y2']), (0, 255, 0), 3)
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
    
    return img

def collect_channel_data(channel_data, img, bbox, roi_percentage = DEFAULT_ROI_PERCENTAGE):
    """
    Populate the given dictionary by appending a spatial average for each channel
    within a percentage of the ROI found for the face. The reason for collecting a percentage
    as sometimes the outer portions of the ROI contain hair, eyebrows, etc.
    """

    # get channels from image which is cropped by the bbox produced which is shrunk by 1 - roi_percentage
    bbox, (rc, gc, bc) = _get_cropped_channels(img, bbox, roi_percentage)

    # for each channel, append its mean to the channel dict
    channel_data['r'].append(np.mean(rc))
    channel_data['g'].append(np.mean(gc))
    channel_data['b'].append(np.mean(bc))

    return bbox

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
    return {
        'x1': xi,
        'y1': yi,
        'x2': xi + xw,
        'y2': yi + yh
    }, cv2.split(cropped_img)

def _get_forehead_bbox(vid, W = 100, resize = False):

    # search the first W frames for a face
    bbox_dict = None
    for i in tqdm(range(W)):
        
        success, frame = vid.read()
        if not success:
            print("Video ended before face was found!")
            return None

        img = process_img(frame, resize = resize)

        # load facial image recognition class and get initial bounding box
        imgProcessing = FacialImageProcessing(False)
        bounding_boxes, _ = imgProcessing.detect_faces(img)

        if len(bounding_boxes) > 0:
            bbox = bounding_boxes[0]
            bbox_dict = {'x1': bbox[0],'y1': bbox[1],'x2': bbox[2],'y2': bbox[3]}
            print("Bounding Boxes: {0}".format(bbox_dict))
        else:
            continue

        # adjust y2 value so that box encloses just the user's forehead
        try:
            bbox_dict['y2'] -= find_forehead(img, bbox_dict)
            return bbox_dict, success, frame
        except: 
            continue

    if bbox_dict is None:
        print(f'Face not found in first {W} frames.')
        return None, False, None
    else:
        return bbox_dict, success, frame


def process_img(frame, resize = False):

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

def display_frame(img, bbox = None):
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
    
    # IMG SHAPE:  (720, 1280, 3)
    # track_video()

    # from laptop IMG SHAPE: (720, 1080, 3)
    # track_video('/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/Movie on 2-2-23 at 3.31 PM.mp4')
    
    # from pixel IMG SHAPE:  (512, 288, 3)
    track_video('/Users/samuelhmorton/indiv_projects/school/masters/pydroid/example/android/app/src/main/python/PXL_20230202_212010481.mp4')