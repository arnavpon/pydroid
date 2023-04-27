import cv2
import dlib
import os
import numpy as np
import json
from datetime import datetime

from forehead import find_forehead
from FaceDetection_MT1 import (
    FacialImageProcessing,
    TAKEN_W, TAKEN_H,
    LOADED_W, LOADED_H,
    IMG_THRESH
)
from pipeline_dev.pipeline_v1.tracking import track_video


def main(arguments):
    """
    Cleaned up main method for face detection with hardcoded dimensions for images based on the specific
    Google Pixel that I'm using. That will need to be improved in the future.
    """

    print(f'[python] Main method of face_detection script...')
    vid_path = arguments.get("vid_path")
    print('Tracking the vid path')
    track_video(vid_path)
    
    
    
    # vid = cv2.VideoCapture(vid_path)

    # success, frame = vid.read()

    # # initialize bbox_dict to failure state
    # bbox_dict = {'x1': 0.0,'y1': 0.0,'x2': 0.0,'y2': 0.0}

    # # loop through the frames until a face is found
    # while success:
        
    #     img = np.copy(frame)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = np.swapaxes(img, 0, 1)
    
    #     imgProcessing = FacialImageProcessing(False)
    #     bounding_boxes, _ = imgProcessing.detect_faces(img)

    #     if len(bounding_boxes) > 0:
            
    #         bbox = bounding_boxes[0]
    #         bbox_dict = {'x1': bbox[0],'y1': bbox[1],'x2': bbox[2],'y2': bbox[3]}
    #         print(f'Bounding Boxes: {bbox_dict}')

    #         # adjust y2 value so that box encloses just the user's forehead
    #         #bbox_dict['y2'] -= find_forehead(img, bbox_dict)

    #         return json.dumps(bbox_dict)
        
    #     else:
    #         print('No face detected!')
    #         bbox_dict = {'x1': 0.0,'y1': 0.0,'x2': 0.0,'y2': 0.0}

    #         success, frame = vid.read()

    # print('Went through each frame so returning failure')
    # return bbox_dict