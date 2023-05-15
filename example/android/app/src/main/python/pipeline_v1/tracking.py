import cv2
from datetime import datetime
import dlib
import numpy as np
import pandas as pd
import os

from forehead import find_forehead
from FaceDetection_MT1 import (
    FacialImageProcessing,
    TAKEN_W, TAKEN_H,
    LOADED_W, LOADED_H,
    IMG_THRESH
)

PCA_COMPS = 3
DEFAULT_ROI_PERCENTAGE = 0.25
DEFAULT_CSV_NAME = './channels.csv'

def track_video(video_path = None, channel_filepath = DEFAULT_CSV_NAME, color_filter = cv2.COLOR_BGR2RGB,
                resize_img = False, S = None, face_renew = None, use_noise = False, use_dim = False, use_bright = False):
    """
    Based on: https://xailient.com/blog/learn-how-to-create-a-simple-face-tracking-system-in-python/
    """

    fn = 0

    # load video with cv2
    if video_path is None:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames:", total_frames)

    bbox, (success, frame) = _get_forehead_bbox(
        video,
        color_filter,
        resize = resize_img,
        use_noise = False,
        use_dim = False,
        use_bright = False,
        fn = fn
    )
    if bbox is None:
        print('[Python] Face not found.')
        return
    
    tracker = _init_correlation_tracker(frame, bbox)
    if face_renew is not None:
        frames_till_renewal = face_renew - 1

    channel_data = {k: [] for k in ['r', 'g', 'b']}
    start_time = datetime.today()

    print('[Python] Tracking face...')
    while success:
        print(f'[Python] Frame: {fn}', end = '\r')
        
        if S is not None and (datetime.today() - start_time).seconds > S:
            break
    
        if face_renew is None:
            success, frame = _vid_read(
                video,
                use_noise = use_noise if fn > 1000 and fn < 4000 else False,
                use_dim = use_dim if fn > 1000 and fn < 4000 else False,
                use_bright = use_bright if fn > 1000 and fn < 4000 else False,
                fn = fn
            )

            # cv2.imshow('Image Title', frame)

            # # Wait for a key press and close the window
            # cv2.waitKey(0)

            fn += 1
        
        else:
            
            if frames_till_renewal == 0:
                bbox, (success, frame) = _get_forehead_bbox(
                    video,
                    resize = resize_img,
                    use_noise = False,
                    use_dim = False,
                    use_bright = False,
                )
                if bbox is not None:
                    tracker = _init_correlation_tracker(frame, bbox)
                frames_till_renewal = face_renew - 1
            
            else:
                frames_till_renewal -= 1

        
        if success:
            curr_img = _track(frame, tracker, channel_data, color_filter, resize_img)
            if curr_img is None:
                continue
    
    data = pd.DataFrame(data = channel_data)
    if channel_filepath is not None:
        data.to_csv(channel_filepath, index = False)
    video.release()
    return data

def _init_correlation_tracker(frame, bbox):
    """
    Initialize the correlation tracker
    """

    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2']))
    tracker.start_track(frame, rect)
    return tracker


def _track(frame, tracker, channel_data, color_filter, resize_img):

    img = process_img(frame, resize = resize_img)
    
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

    if condensed_bbox is None:
        return condensed_bbox

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
    bbox, channels = _get_cropped_channels(img, bbox, roi_percentage)

    if len(channel_data.keys()) != len(channels):
        return None

    for k, c in zip(channel_data.keys(), channels):
        channel_data[k].append(np.mean(c))

    return bbox

def _get_cropped_channels(img, bbox, roi_percentage):

    roi_width = bbox['x2'] - bbox['x1']
    roi_height = bbox['y2'] - bbox['y1']

    xi = max(0, int(bbox['x1'] + ((roi_width * (1 - roi_percentage)) / 2)))
    xw = int(roi_width * roi_percentage)

    yi = max(0, int(bbox['y1'] + ((roi_height * (1 - roi_percentage)) / 4)))
    yh = int(roi_height * roi_percentage)
    

    cropped_img = img[yi: yi + yh, xi: xi + xw]

    channels = cv2.split(cropped_img)
    return {
        'x1': xi,
        'y1': yi,
        'x2': xi + xw,
        'y2': yi + yh
    }, channels

def _get_forehead_bbox(vid, color_filter, iter_lim = 300, resize = False, use_noise = False, use_dim = False, use_bright = False, fn = 0):

    # search the first iter_lim frames for a face
    bbox_dict = None
    for i in range(iter_lim):
        
        success, frame = _vid_read(vid, use_noise = use_noise, use_dim = use_dim, use_bright = use_bright, fn = fn)
        if not success:
            print("Video ended before face was found!")
            return None

        img = process_img(frame, resize = resize)
        imgProcessing = FacialImageProcessing(False)
        bounding_boxes, _ = imgProcessing.detect_faces(img)

        if len(bounding_boxes) == 0:
            continue

        bbox = bounding_boxes[0]
        bbox_dict = {'x1': bbox[0],'y1': bbox[1],'x2': bbox[2],'y2': bbox[3]}

        try:
            bbox_dict['y2'] -= find_forehead(img, bbox_dict)
            return bbox_dict, (success, img)
        except Exception as e:
            print(e) 
            continue

    if bbox_dict is None:
        print(f'Face not found in first {iter_lim} frames.')
        return None, (False, None)
    else:
        return bbox_dict, (success, img)


def process_img(frame, resize = False):

    # convert the image to RGB
    img = frame.copy()

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


def _vid_read(vid, use_noise = False, use_dim = False, use_bright = False, fn = 0):
    success, frame = vid.read()
    if not success:
        return success, frame

    fn += 1
    if use_noise:
        frame = _add_gaussian_noise(frame, mean = 0, stddev = 60)
    
    if use_dim:
        frame = _adjust_brightness(frame, delta = -60)
    
    if use_bright:
        frame = _adjust_brightness(frame, delta = 60)

    return success, frame


def _add_gaussian_noise(image, mean = 0, stddev = 30):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, stddev, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + gauss
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def _adjust_brightness(image, delta):
    return np.clip(image + delta, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    import os
    os.mkdir('channel_data3_bright')
    for subject in range(1, 8):
        sid = f'00{subject}'
        print('subject', sid)

        # change the path to make it work
        path = f'/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_{sid}/trial_001/video/video.MOV'
        track_video(
            video_path = path,
            channel_filepath = f'channel_data3_bright/ieee-subject-{sid}-trial-001-mean-0-std-30.csv',
            use_noise=False,
            use_dim=False,
            use_bright=True
        )
