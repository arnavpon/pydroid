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

    ys = []
    for face in face_landmarks:
        return face['nose_bridge'][-1][1]
        # print(face['nose_bridge'])
        # for pos in face['nose_bridge']: ys.append(pos[1])
        # for pos in face['right_eyebrow']: ys.append(pos[IDX])
    
    print('Retuning ', int(sum(ys) / len(ys)))
    return int(sum(ys) / len(ys))


def _crop_image(img, bbox):

    x = bbox['x1']
    y = bbox['y1']
    w = abs(bbox['x2'] - x)
    h = abs(bbox['y2'] - y)

    return img[int(y): int(y + h), int(x): int(x + w)]
