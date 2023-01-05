"""
Module for segmenting forehead location using a precomputed bounding box
telling us the location of the user's face in an image.
"""

import face_recognition


def find_forehead(img, bbox):

    # get a cropped version of the image using the bounding box
    cropped = _crop_image(img, bbox)

    # convert cropped image to the correct storage of RGB values
    # source: https://github.com/ageitgey/face_recognition/issues/441
    img_for_fr = cropped[:, :, ::-1]

    # find facial features for the face in the image
    face_landmarks = face_recognition.face_landmarks(img_for_fr)

    # return the y position of the top of the nose bridge
    # we can then use this value to adjust the bounding box
    # to enclose just the user's forehead
    face = list(face_landmarks.keys())[0]
    return face['nose_bridge'][-1][1]

def _crop_image(img, bbox):
    """
    Given an image and bounding box, return a version of the image
    cropped to just include the bounding box.
    """

    x = bbox['x1']
    y = bbox['y1']
    w = abs(bbox['x2'] - x)
    h = abs(bbox['y2'] - y)

    return img[int(y): int(y + h), int(x): int(x + w)]
