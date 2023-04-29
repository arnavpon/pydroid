"""
Example module for processing the IEEE data, taken from
https://github.com/mxahan/project_rppg/blob/master/Codes/Data_read_only/data_read_main.py
"""

import os
import numpy as np
import cv2
import glob
import pandas as pd

path_dir = '../../../../Dataset/Personal_collection/MPSC_rppg/subject_001/trial_001/video/'

ppgtotal =  pd.read_csv(path_dir +'../empatica_e4/BVP.csv')
EventMark = pd.read_csv(path_dir+'../empatica_e4/tags.csv')

dataPath = os.path.join(path_dir, '*.MOV')

files = glob.glob(dataPath)  # care about the serialization
# end load pathdir
list.sort(files) # serialing the data

# Take time stamp and multiple by 64. Take starting time of the BVP file, 
# subtract the tags.csv from the BVP start time, multiply by 64 to get the sample number. x



data = []
im_size = (100,100)

cap = cv2.VideoCapture(files[0])

import pdb

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    gray  = gray[:,:,1]
    #update the following line
    gray =  gray[10:900, 720:1500]
    
    gray = cv2.resize(gray, im_size)
    # pdb.set_trace()
    data.append(gray)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


fps = cap.get(cv2.CAP_PROP_FPS)
    
cap.release()
cv2.destroyAllWindows()
data =  np.array(data)

# The starting points are the crucial, 
# this section needs select both the sratrting of video and the ppg point
# check fps and starting time in BVP.csv
# Match the lines from the supplimentary text file for the data

evmarknp =  EventMark.to_numpy()
ppgnp =  ppgtotal.to_numpy()

#update the following line/s

start_gap =  evmarknp[0] - 1593893213 
end_point =  evmarknp[1] - evmarknp[0] # default (..[1] -..[0])


data_align = data[307:307+np.int(end_point*30)+5]  ### Contains the video
ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]  ### contains the aligned PPG