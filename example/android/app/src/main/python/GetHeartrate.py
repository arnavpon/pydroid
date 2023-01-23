import cv2
import dlib
import os
import numpy as np

from forehead import find_forehead
from FaceDetection_MT1 import (
    FacialImageProcessing,
    TAKEN_W, TAKEN_H,
    LOADED_W, LOADED_H,
    IMG_THRESH
)

def _process_img(img, resize = False):

    # convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize image differently depending on if it's taken locally (<IMG_THRESH) or loaded from the library
    if resize:
        img = (
            cv2.resize(img, (int(TAKEN_W), int(TAKEN_H)), interpolation = cv2.INTER_AREA)
            if img.shape[0] < IMG_THRESH
            else cv2.resize(img, (int(LOADED_W), int(LOADED_H)), interpolation = cv2.INTER_AREA)
        )

    return img


def _find_forehead(img, resize = False):

    img = _process_img(img, resize = resize)

    # load facial image recognition class and get initial bounding box
    imgProcessing = FacialImageProcessing(False)
    bounding_boxes, _ = imgProcessing.detect_faces(img)

    if len(bounding_boxes) > 0:
        bbox = bounding_boxes[0]
        bbox_dict = {'x1': bbox[0],'y1': bbox[1],'x2': bbox[2],'y2': bbox[3]}
        print("Bounding Boxes: {0}".format(bbox_dict))
    else:
        print("No face detected!")
        bbox_dict = {'x1': 0.0,'y1': 0.0,'x2': 0.0,'y2': 0.0}
        return bbox_dict

    # adjust y2 value so that box encloses just the user's forehead
    bbox_dict['y2'] -= find_forehead(img, bbox_dict)

    return bbox_dict


def main(arguments):

    print(f'[python] Analyzing video')
    
    # models_path, _ = os.path.split(os.path.realpath(__file__))
    # model_file = os.path.join(models_path, 'shape_predictor_68_face_landmarks.dat')
    # if not os.path.exists(model_file):
    #     raise Exception("no model found at specified path")

    # # initialize the model for face landmark detection
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(model_file)

    # load the video
    print('loading the video')
    vid_path = arguments.get("vid_path")
    video = cv2.VideoCapture(vid_path)

    success, frame = video.read()

    if success:
        print(f'Frame starts at {frame.shape}')
        bbox = _find_forehead(frame, resize = False)
        print(f'bbox: {bbox}')
    else:
        print('Could not read image')
    # while success:
        
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #     # resize image differently depending on if it's taken locally (<IMG_THRESH) or loaded from the library
    #     frame = cv2.resize(frame, (int(TAKEN_W), int(TAKEN_H)), interpolation = cv2.INTER_AREA)
    #     # img = (
    #         # cv2.resize(img, (int(TAKEN_W), int(TAKEN_H)), interpolation = cv2.INTER_AREA)
    #         # if img.shape[0] < IMG_THRESH
    #         # else cv2.resize(img, (int(LOADED_W), int(LOADED_H)), interpolation = cv2.INTER_AREA)
    #     # )
    #     sam = [list(l) for l in frame]
    #     print('img shape', frame.shape, type(frame))
    #     print(sam)


    #     # convert the image color to grayscale
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    #     # detect the face
    #     faces = detector(gray, 1)
    #     # faces = detector(frame, 1)
    #     lands = []
    #     for face in faces:
    #         print('in here')
    #         # get landmarks
    #         landmarks = predictor(gray, face)
    #         print('got landmarks', type(landmarks))

    #         shape_np = np.zeros((68, 2), dtype = int)
    #         for i in range(0, 68):
    #             shape_np[i] = (landmarks.part(i).x, landmarks.part(i).y)
    #         landmarks = shape_np

    #         lands.append(landmarks)

    #     print('Printing landmarks')
    #     if len(lands) > 0:
    #         print(len(lands))
    #         print(len(lands[0]))
    #         print(lands[0])
    #     print('got here')

    #     success, frame = video.read()
    
    video.release()

    return True
