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

    print(f'[python] Main method of face_detection script...')
    vid_path = arguments.get("vid_path")
    vid = cv2.VideoCapture(vid_path)

    success, frame = vid.read()

    if success:
        
        img = np.copy(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sam = np.swapaxes(np.copy(frame), 0, 1)
        print(f'new shape: {sam.shape}')

        print(f'img is {img.shape}')
    
        imgProcessing = FacialImageProcessing(False)
        bounding_boxes, _ = imgProcessing.detect_faces(sam)

        if len(bounding_boxes) > 0:
            bbox=bounding_boxes[0]
            bbox_dict = {'x1': bbox[0],'y1': bbox[1],'x2': bbox[2],'y2': bbox[3]}
            print("Bounding Boxes: {0}".format(bbox_dict))
        else:
            print("No face detected!")
            bbox_dict = {'x1': 0.0,'y1': 0.0,'x2': 0.0,'y2': 0.0}

        # adjust y2 value so that box encloses just the user's forehead
        bbox_dict['y2'] -= find_forehead(sam, bbox_dict)

        return json.dumps(bbox_dict)  # returns value as JSON object
    
    else:
        print('Couldnt even read the frame')
        return json.dumps({'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0})
