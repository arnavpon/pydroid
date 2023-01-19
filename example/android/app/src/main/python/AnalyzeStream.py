import cv2
import os
import PIL
from PIL import Image
import imghdr
import json
import pickle

from tracking import track_image


def main(arguments):
    print('in the script')

    # load image from the given path
    img_path = arguments.get('img_path')
    img = cv2.imread(img_path)

    print('in here')

    # check if correlation tracker given
    tracker_path = arguments.get('tracker_path')
    if tracker_path != '':
        tracker_file = open(tracker_path, 'r')
        tracker = pickle.load(tracker_file)
        tracker_file.close()
    else: tracker = None

    print('GOT HERE')
    print(img.shape)
    print(tracker)
    print('==End')
    
    bbox = track_image(img, tracker)
    return json.dumps(bbox)
