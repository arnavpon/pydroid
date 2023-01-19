import cv2
import dlib
import numpy as np
import pickle

from forehead import find_forehead
from FaceDetection_MT1 import (
    FacialImageProcessing,
    TAKEN_W, TAKEN_H,
    LOADED_W, LOADED_H,
    IMG_THRESH
)

def track_image(img, tracker):

    if tracker is None:

        # get bbox for forehead
        bbox = _find_forehead(img, resize = True)
        print('bbox is')
        print(bbox)

        # init tracker to track forehead in future frames
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2']))
        tracker.start_track(img, rect)

        # save tracker locally
        tracker_file = open('tracker.sav', 'w')
        pickle.dump(tracker, tracker_file)
        tracker_file.close()

        return bbox
    
    else:
        print('tracker isnt none')
        return _track_img(img, tracker, tracker_path, True)


def track_video(video_path = None):
    """
    Based on: https://xailient.com/blog/learn-how-to-create-a-simple-face-tracking-system-in-python/
    """

    # load video with cv2
    if video_path is None:
        video = cv2.VideoCapture(0) # this is the magic!
    else:
        video = cv2.VideoCapture(video_path)
    
    resize_img = video_path is not None

    # read first frame of the video
    success, frame = video.read()

    # get bounding box for forehead from first frame
    if success:
        bbox = _find_forehead(frame, resize = resize_img)
    else:
        print('[Python] Face not found in first frame')
        return
    
    # init the tracker
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2']))
    tracker.start_track(frame, rect)
    
    # display the first frame
    _display_frame(frame, bbox)

    while success:
        
        # read the next frame
        success, frame = video.read()

        if success:

            # get new bbox for the current frame
            curr_img = _track(frame, tracker, resize_img)

            # display current frame
            _display_frame(curr_img)

            if cv2.waitKey(10) == 27:
                break
    
    video.release()


def _track(frame, tracker, resize_img):

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

    # draw new rectangle on the img
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
    
    return img

def _track_img(frame, tracker, tracker_path, resize_img):

    img = _process_img(frame, resize = resize_img)

    # make a copy of the frame
    img = np.copy(frame)
    
    # update the tracker based on the current image
    tracker.update(img)
    pos = tracker.get_position()

    # save the updated tracker
    tracker_file = open(tracker_path, 'w')
    pickle.dump(tracker, tracker_file)
    tracker_file.close()

    # isolate starting and ending x/y
    return {
        'x1': int(pos.left()),
        'y1': int(pos.top()),
        'x2': int(pos.right()),
        'y2': int(pos.bottom())
    }

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
    track_video('./PXL_20220724_101216902.mp4')
    # track_video()
