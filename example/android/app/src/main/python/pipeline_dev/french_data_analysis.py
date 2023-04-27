import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tracking import track_video
from pipeline import pipeline

# dataset folder
root = 'validation_data/french_data/DATASET_1/'

# get folder list
dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

# Iterate through all directories
for i in range(len(dirs)):
    print(f'DIR: {dirs[i]}')
    vidFolder = os.path.join(root, dirs[i])
    
    # load ground truth
    gtfilename = os.path.join(vidFolder, 'gtdump.xmp') # DATASET_1
    if os.path.isfile(gtfilename):
        gtdata = np.loadtxt(gtfilename, delimiter=',')
        gtTrace = gtdata[:, 3]
        gtTime = gtdata[:, 0] / 1000
        gtHR = gtdata[:, 1]
    else:
        gtfilename = os.path.join(vidFolder, 'ground_truth.txt') #DATASET_2
        if os.path.isfile(gtfilename):
            gtdata = np.loadtxt(gtfilename, delimiter=',')
            gtTrace = gtdata[0, :].reshape(-1, 1)
            gtTime = gtdata[2, :].reshape(-1, 1)
            gtHR = gtdata[1, :].reshape(-1, 1)
    
    # normalize data (zero mean and unit variance)
    gtTrace = (gtTrace - np.mean(gtTrace)) / np.std(gtTrace)

    print('Ground truth: ', np.mean(gtHR))
    lim = 400
    w = 30
    print('RBG: ', pipeline(f'channel_data/french-{dirs[i]}-rgb.csv', moving_average_window = w, lim = lim, with_plots = False))
    print('YUV: ', pipeline(f'channel_data/french-{dirs[i]}-yuv.csv', moving_average_window = w, lim = lim, with_plots = True))
    print()

    # vfile = os.path.join(vidFolder, 'vid.avi')

    # track_video(
    #     vfile, f'./channel_data/french-{dirs[i]}-rgb.csv', show_frames = True
    # )
    # track_video(
    #     vfile, f'./channel_data/french-{dirs[i]}-yuv.csv', show_frames = True, color_filter = cv2.COLOR_BGR2YUV
    # )



    
    # open video file
    # vidObj = cv2.VideoCapture(os.path.join(vidFolder, 'vid.avi'))
    # fps = vidObj.get(cv2.CAP_PROP_FPS)
    
    # n = 0
    # while True:
    #     # track frame index
    #     n += 1
        
    #     # read frame by frame
    #     ret, img = vidObj.read()
    #     if not ret:
    #         break
        
    #     # perform operations on frame
    #     cv2.imshow('frame', img)
    #     cv2.waitKey(1)
    
    # print(n, len(gtTrace))
    
    # vidObj.release()
    # cv2.destroyAllWindows()
    #print('{}: {} - {}; {} - {}'.format(i, n, len(gtTime), vidObj.get(cv2.CAP_PROP_POS_MSEC), gtTime[-1]))
