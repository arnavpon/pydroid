import cv2
from datetime import datetime
import dlib
import numpy as np

NUM_LANDMARKS = 68
NUM_FIXED = 17
OUTSIDE_LANDMARK = 3
INSIDE_LANDMARK = 30

# initialize model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# load video locally for testing
vid = cv2.VideoCapture('PXL_20220724_101216902.mp4')

def _get_translation(shape, first_frame_locs):

    diff_sum = 0
    for i in range(NUM_FIXED):
        diff = np.array(first_frame_locs[i]) - np.array(shape[i])
        norm2 = pow(np.linalg.norm(diff), 2)
        diff_sum += norm2
    
    return np.ones((2, 2)), np.ones((2, 1))

start = datetime.today()

success, frame = vid.read()
first_frame_locs = {}
while success:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = np.zeros((68, 2), dtype = int)
        
        for i in range(0, NUM_LANDMARKS):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        for i, (x, y) in enumerate(shape[0: NUM_FIXED]):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    
    if len(first_frame_locs) == 0:
        for i in range(NUM_FIXED):
            first_frame_locs[i] = shape[i]
    else:
        A, b = _get_translation(shape, first_frame_locs)
    
    # determine poi
    poix = (shape[OUTSIDE_LANDMARK][0] + shape[INSIDE_LANDMARK][0]) / 2
    poiy = (shape[OUTSIDE_LANDMARK][1] + shape[INSIDE_LANDMARK][1]) / 2
    poi = np.array([poix, poiy])

    # transform poi
    poi_T = (A * poi) + b

	
    cv2.imshow('Landmark Detection', frame)

    if cv2.waitKey(10) == 27:
        break

    success, frame = vid.read()

print(f'Run in {datetime.today() - start}.')
vid.release()
