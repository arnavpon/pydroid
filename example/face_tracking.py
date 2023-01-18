import cv2
from datetime import datetime
import dlib
import numpy as np

NUM_LANDMARKS = 68
NUM_FIXED = 17
OUTSIDE_LANDMARK = 3
INSIDE_LANDMARK = 30


# load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

    # Convert the image color to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the face
    rects = detector(gray, 1)
    # Detect landmarks for each face
    for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)
	# Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
	    # Draw the circle to mark the keypoint 
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
		
    # Display the image
    cv2.imshow('Landmark Detection', frame)

    # Press the escape button to terminate the code
    if cv2.waitKey(10) == 27:
        break

    success, frame = vid.read()

print(f'Run in {datetime.today() - start}.')
vid.release()
