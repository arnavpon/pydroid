import cv2
import os
import PIL
from PIL import Image
import imghdr
import json


def main(arguments):

    img_path = arguments.get("img_path")
    print("THE PATH", img_path, os.path.exists(img_path))
    img = cv2.imread(img_path)
    # img2 = io.imread(img_path)
    print('PRIngitng the img')
    print(f'SAM: {type(img)}, {img.shape}')
    print(img)

    try:
        if os.path.exists(img_path): print('It exists')
        im = Image.open(img_path)
        im.verify() #I perform also verify, don't know if he sees other types o defects
        im.close() #reload is necessary in my case
        # im = Image.load(img_path) 
        # im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        # im.close()
    except Exception as e: 
        print(img.what(img_path))
        print(f'[Python] BROKEN: {e}')

    # print('printing from plt')
    # print(img2)

    print(f'Img shape: {img.shape}')
    bbox = {'x1': img.shape[0] / 4, 'y1': img.shape[1] / 4, 'x2': (3 * img.shape[0]) / 4, 'y2': (3*img.shape[1])/4}
    return json.dumps(bbox)
