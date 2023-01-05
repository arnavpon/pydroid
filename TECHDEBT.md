## Doc Explanation
As tech debt piles up, it'll be a bit easier to address it if I've made note of debt as it accumulates.

### Debt 1: Image sizing
In order to get boxes to show up in the correct place on the user's face in the app, I needed to hardcode
values in FaceDetection_MT1.py for the cases of the user taking a screenshot directly in the app vs. loading
an image from their library. I used these values to resize the images for face detection. The reason why
the boxes were originally being drawn incorrectly was that the image size was getting lost in translation
between Flutter and Python.

### Debt 2: Forehead detection from detected face
After we detect the user's face, I crop the photo based on the returned bounding box and then get the location
of the user's forehead. However, the library I used (face_detection) doesn't actually support detecting the forehead.
It does support detecting the "nose bridge", so I use that and consider the top of the nose bridge to be the bottom
of the face.