import cv2
from datetime import datetime
import dlib
import numpy as np

NUM_LANDMARKS = 68

# initialize model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# load video locally for testing
vid = cv2.VideoCapture('PXL_20220724_101216902.mp4')

start = datetime.today()

success, frame = vid.read()
while success:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = np.zeros((68, 2), dtype = int)
        
        for i in range(0, NUM_LANDMARKS):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        for i, (x, y) in enumerate(shape):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
		
    cv2.imshow('Landmark Detection', frame)

    if cv2.waitKey(10) == 27:
        break

    success, frame = vid.read()

print(f'Run in {datetime.today() - start}.')
vid.release()
