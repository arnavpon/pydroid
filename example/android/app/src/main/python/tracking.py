import cv2
import dlib
import numpy as np

from forehead import find_forehead
from FaceDetection_MT1 import (
    FacialImageProcessing,
    TAKEN_W, TAKEN_H,
    LOADED_W, LOADED_H,
    IMG_THRESH
)

def track_video(video_path):
    """
    Based on: https://xailient.com/blog/learn-how-to-create-a-simple-face-tracking-system-in-python/
    """

    # load video with cv2
    video = cv2.VideoCapture(video_path)

    # read first frame of the video
    success, frame = video.read()

    first_frame_img = np.copy(frame)  # make a copy of the frame for manipulation

    # get bounding box for forehead from first frame
    if success:
        bbox = _find_forehead(frame)
    else:
        print('[Python] Face not found in first frame')
        return
    
    # init the tracker
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2']))
    tracker.start_track(frame, rect)
    
    # display the first frame
    frames = []
    _display_frame(frame, bbox)
    frames.append(frame)

    while success:
        
        # read the next frame
        success, frame = video.read()

        if success:

            # get new bbox for the current frame
            curr_img = _track(frame, tracker)

            # display current frame
            _display_frame(curr_img)
            frames.append(frame)

            if cv2.waitKey(10) == 27:
                break
    
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 1080))
    for frame in frames:
        out.write(frame) # frame is a numpy.ndarray with shape (1280, 720, 3)
    out.release()
    video.release()


def _track(frame, tracker):

    img = _process_img(frame)

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

    # draw new rectangle on the img
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
    
    return img


def _find_forehead(img):

    img = _process_img(img)

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

    # adjust y2 value so that box encloses just the user's forehead
    bbox_dict['y2'] -= find_forehead(img, bbox_dict)

    return bbox_dict


def _process_img(img):

    # convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize image differently depending on if it's taken locally (<IMG_THRESH) or loaded from the library
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
    
    print(img)
    cv2.imshow('Image', img)


if __name__ == '__main__':
    track_video('./PXL_20220724_101216902.mp4')
