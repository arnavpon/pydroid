import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)

success, frame = cap.read()

while success:   

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for rect in rects:

        shape = predictor(gray, rect)

        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
	    # Draw the circle to mark the keypoint 
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		
    # Display the image
    print(ret)
    cv2.imshow('Landmark Detection', image)

    success, frame = cap.read()

    # Press the escape button to terminate the code
    if cv2.waitKey(10) == 27:
        break

cap.release()