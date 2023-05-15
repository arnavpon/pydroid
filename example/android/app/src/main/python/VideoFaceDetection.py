import cv2
import json
import numpy as np

# local modules
from ml_pipeline_basic import pipeline
from pipeline_v1.tracking import track_video


def main(arguments):
    """
    Cleaned up main method for face detection with hardcoded dimensions for images based on the specific
    Google Pixel that I'm using. That will need to be improved in the future.
    """

    print(f'[python] Main method of HR estimation script...')
    
    # load the video
    vid_path = arguments.get("vid_path")
    rgb = track_video(vid_path, channel_filepath = None)
    res = pipeline(rgb)
    return json.dumps(res)
