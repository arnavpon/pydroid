import cv2
import dlib
import os
import numpy as np
import json

from forehead import find_forehead
from FaceDetection_MT1 import (
    FacialImageProcessing,
    TAKEN_W, TAKEN_H,
    LOADED_W, LOADED_H,
    IMG_THRESH
)


def main(arguments):
    """
    Cleaned up main method for face detection with hardcoded dimensions for images based on the specific
    Google Pixel that I'm using. That will need to be improved in the future.
    """

    print(f'[python] Main method of HR estimation script...')
    
    # load the video
    vid_path = arguments.get("vid_path")
    vid = cv2.VideoCapture(vid_path)

    # read the initial frame
    success, frame = vid.read()

    # loop through the frames until a face is found
    found_face = False
    nframes = 0
    while success:
        nframes += 1

        img = _process_img(frame)
    
        imgProcessing = FacialImageProcessing(False)
        bounding_boxes, _ = imgProcessing.detect_faces(img)

        if len(bounding_boxes) > 0:
            
            bbox = bounding_boxes[0]
            bbox_dict = {'x1': bbox[0],'y1': bbox[1],'x2': bbox[2],'y2': bbox[3]}
            print(f'Bounding Boxes: {bbox_dict}')

            # move onto the next frame
            success, frame = vid.read()

            found_face = True
            break
        
        else:
            print('No face detected!')
            success, frame = vid.read()

    
    if not found_face:
        print(f'No frames found in {nframes} frames')
        return json.dumps({'x1': 0.0,'y1': 0.0,'x2': 0.0,'y2': 0.0})
    

    # init array to track all bounding bounding boxes
    # it starts out already containing the box for the first frame that a face was found
    boxes = [bbox_dict]

    # init dlib correlation tracker
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(
        int(bbox_dict['x1']),
        int(bbox_dict['y1']),
        int(bbox_dict['x2']),
        int(bbox_dict['y2'])
    )
    tracker.start_track(frame, rect)

    # get boxes for the remaining frames in the video
    while success:
        nframes += 1

        curr_box = _track(frame, tracker)
        if curr_box is not None:
            boxes.append(curr_box)

        success, frame = vid.read()
    
    print(f'Returning {len(boxes)} boxes out of {nframes} frames')
    return json.dumps(boxes)


def _process_img(frame):
    img = np.copy(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.swapaxes(img, 0, 1)
    return img


def _track(frame, tracker):

    # process the new image
    img = _process_img(frame)
    
    # update the tracker based on the current image
    tracker.update(img)
    pos = tracker.get_position()

    # return a box with the new position of the rectangle
    return {
        'x1': int(pos.left()),
        'y1': int(pos.top()),
        'x2': int(pos.right()),
        'y2': int(pos.bottom())
    }
