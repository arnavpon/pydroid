import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from scipy.io import loadmat
import random
from random import seed, randint
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
import pdb
import pickle

print('Doing subject 1')

subject = '001'
path_dir = f'../../../../../../../project_rppg/Codes/subject_{subject}/trial_001/'
ppgtotal =  pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_001/trial_001/empatica_e4/BVP.csv')
EventMark = pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_001/trial_001/empatica_e4/tags.csv')
dataPath = os.path.join(f'{path_dir}video', '*.MOV')
files = glob.glob(dataPath)
list.sort(files)


data = []
im_size = (100,100)
cap = cv2.VideoCapture(files[0])
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray  = gray[:,:,1]
    gray =  gray[10:900, 720:1500]
    gray = cv2.resize(gray, im_size)
    data.append(gray)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
cv2.destroyAllWindows()

data = np.array(data)
evmarknp =  EventMark.to_numpy()
ppgnp =  ppgtotal.to_numpy()
start_gap =  evmarknp[0] - 1593893213
end_point =  evmarknp[1] - evmarknp[0]
ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]
data_align = data[307:307+np.int(end_point*30)+5]

random.seed(1)
rv = np.arange(0,5000, 1)+1000
np.random.shuffle(rv)
rv =  np.array(rv)
pulR = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1])

trainX = []
trainY = []
data_align = data_align[:,:,:,np.newaxis]
frame_cons = 40 # how many frame to consider at a time

for j, i in enumerate(rv):
    
    img = np.reshape(data_align[i:i+frame_cons,:,:,0], [frame_cons, *im_size])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    p_point = np.int(np.round(i*64/30))
    ppg = pulR[p_point: p_point+85, 0]
    trainY.append(ppg)


ddir = 'replicate_pickle_data'
ssub = 'sub1'
if not os.path.exists(ddir): os.mkdir(ddir)
if not os.path.exists(f'{ddir}/{ssub}'): os.mkdir(f'{ddir}/{ssub}')

with open(f'{ddir}/{ssub}/data_align.pkl', 'wb') as f:
    pickle.dump(data_align, f)
f.close()
with open(f'{ddir}/{ssub}/ppgnp_align.pkl', 'wb') as f:
    pickle.dump(ppgnp_align, f)
f.close()
with open(f'{ddir}/{ssub}/trainX.pkl', 'wb') as f:
    pickle.dump(trainX, f)
f.close()
with open(f'{ddir}/{ssub}/trainY.pkl', 'wb') as f:
    pickle.dump(trainY, f)
f.close()
with open(f'{ddir}/{ssub}/pulR.pkl', 'wb') as f:
    pickle.dump(pulR, f)
f.close()

print('Doing subject 2')

subject = '002'
path_dir = f'../../../../../../../project_rppg/Codes/subject_{subject}/trial_001/'
ppgtotal =  pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_002/trial_001/empatica_data/BVP.csv')
EventMark = pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_002/trial_001/empatica_data/tags.csv')
dataPath = os.path.join(f'{path_dir}video', '*.MOV')
files = glob.glob(dataPath) 
list.sort(files)


data = []
im_size = (100,100)
cap = cv2.VideoCapture(files[0])
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray  = gray[:,:,1]
    gray =  gray[0:650, 660:1200]
    gray = cv2.resize(gray, im_size)
    data.append(gray)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
cv2.destroyAllWindows()

data = np.array(data)
evmarknp =  EventMark.to_numpy()
ppgnp =  ppgtotal.to_numpy()
start_gap =  evmarknp[0] - 1594416869
end_point =  evmarknp[1] - evmarknp[0]
ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]
data_align = data[478:478+np.int(end_point*30)+5]

random.seed(1)
rv = np.arange(0,5000, 1)+1000
np.random.shuffle(rv)
rv =  np.array(rv)
pulR = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1])

trainX = []
trainY = []
data_align = data_align[:,:,:,np.newaxis]
frame_cons = 40 # how many frame to consider at a time

for j, i in enumerate(rv):
    
    img = np.reshape(data_align[i:i+frame_cons,:,:,0], [frame_cons, *im_size])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    p_point = np.int(np.round(i*64/30))
    ppg = pulR[p_point: p_point+85, 0]
    trainY.append(ppg)


ddir = 'replicate_pickle_data'
ssub = 'sub2'
if not os.path.exists(ddir): os.mkdir(ddir)
if not os.path.exists(f'{ddir}/{ssub}'): os.mkdir(f'{ddir}/{ssub}')

with open(f'{ddir}/{ssub}/data_align.pkl', 'wb') as f:
    pickle.dump(data_align, f)
f.close()
with open(f'{ddir}/{ssub}/ppgnp_align.pkl', 'wb') as f:
    pickle.dump(ppgnp_align, f)
f.close()
with open(f'{ddir}/{ssub}/trainX.pkl', 'wb') as f:
    pickle.dump(trainX, f)
f.close()
with open(f'{ddir}/{ssub}/trainY.pkl', 'wb') as f:
    pickle.dump(trainY, f)
f.close()
with open(f'{ddir}/{ssub}/pulR.pkl', 'wb') as f:
    pickle.dump(pulR, f)
f.close()

print('Doing subject 3')

subject = '003'
path_dir = f'../../../../../../../project_rppg/Codes/subject_{subject}/trial_001/'
ppgtotal =  pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_003/trial_001/empatica_e4/BVP.csv')
EventMark = pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_003/trial_001/empatica_e4/tags.csv')
dataPath = os.path.join(f'{path_dir}video', '*.MOV')
files = glob.glob(dataPath) 
list.sort(files)

data = []
im_size = (100,100)
cap = cv2.VideoCapture(files[0])
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray  = gray[:,:,1]
    gray =  gray[:, 400:1630]
    gray = cv2.resize(gray, im_size)
    data.append(gray)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
cv2.destroyAllWindows()

data = np.array(data)
evmarknp =  EventMark.to_numpy()
ppgnp =  ppgtotal.to_numpy()
start_gap =  evmarknp[0] - 1594841222
end_point =  evmarknp[1] - evmarknp[0]
ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]
data_align = data[307:307+np.int(end_point*30)+5]

random.seed(1)
rv = np.arange(0,5000, 1)+1000
np.random.shuffle(rv)
rv =  np.array(rv)
pulR = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1])

trainX = []
trainY = []
data_align = data_align[:,:,:,np.newaxis]
frame_cons = 40 # how many frame to consider at a time

for j, i in enumerate(rv):
    
    img = np.reshape(data_align[i:i+frame_cons,:,:,0], [frame_cons, *im_size])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    p_point = np.int(np.round(i*64/30))
    ppg = pulR[p_point: p_point+85, 0]
    trainY.append(ppg)


ddir = 'replicate_pickle_data'
ssub = 'sub3'
if not os.path.exists(ddir): os.mkdir(ddir)
if not os.path.exists(f'{ddir}/{ssub}'): os.mkdir(f'{ddir}/{ssub}')

with open(f'{ddir}/{ssub}/data_align.pkl', 'wb') as f:
    pickle.dump(data_align, f)
f.close()
with open(f'{ddir}/{ssub}/ppgnp_align.pkl', 'wb') as f:
    pickle.dump(ppgnp_align, f)
f.close()
with open(f'{ddir}/{ssub}/trainX.pkl', 'wb') as f:
    pickle.dump(trainX, f)
f.close()
with open(f'{ddir}/{ssub}/trainY.pkl', 'wb') as f:
    pickle.dump(trainY, f)
f.close()
with open(f'{ddir}/{ssub}/pulR.pkl', 'wb') as f:
    pickle.dump(pulR, f)
f.close()

print('Doing subject 4')

subject = '004'
path_dir = f'../../../../../../../project_rppg/Codes/subject_{subject}/trial_001/'
ppgtotal =  pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_004/trial_001/empatica_e4/BVP.csv')
EventMark = pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_004/trial_001/empatica_e4/tags.csv')
dataPath = os.path.join(f'{path_dir}video', '*.MOV')
files = glob.glob(dataPath) 
list.sort(files)

data = []
im_size = (100,100)
cap = cv2.VideoCapture(files[0])
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray  = gray[:,:,1]
    gray =  gray[:, 400:1400]
    gray = cv2.resize(gray, im_size)
    data.append(gray)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
cv2.destroyAllWindows()

data = np.array(data)
evmarknp =  EventMark.to_numpy()
ppgnp =  ppgtotal.to_numpy()
start_gap =  evmarknp[1] - 1595391050
end_point =  evmarknp[2] - evmarknp[1]
ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]
data_align = data[216  : 216 +np.int(end_point*30)+5]

random.seed(1)
rv = np.arange(0,5000, 1)+1000
np.random.shuffle(rv)
rv =  np.array(rv)
pulR = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1])

trainX = []
trainY = []
data_align = data_align[:,:,:,np.newaxis]
frame_cons = 40 # how many frame to consider at a time

for j, i in enumerate(rv):
    
    img = np.reshape(data_align[i:i+frame_cons,:,:,0], [frame_cons, *im_size])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    p_point = np.int(np.round(i*64/30))
    ppg = pulR[p_point: p_point+85, 0]
    trainY.append(ppg)


ddir = 'replicate_pickle_data'
ssub = 'sub4'
if not os.path.exists(ddir): os.mkdir(ddir)
if not os.path.exists(f'{ddir}/{ssub}'): os.mkdir(f'{ddir}/{ssub}')

with open(f'{ddir}/{ssub}/data_align.pkl', 'wb') as f:
    pickle.dump(data_align, f)
f.close()
with open(f'{ddir}/{ssub}/ppgnp_align.pkl', 'wb') as f:
    pickle.dump(ppgnp_align, f)
f.close()
with open(f'{ddir}/{ssub}/trainX.pkl', 'wb') as f:
    pickle.dump(trainX, f)
f.close()
with open(f'{ddir}/{ssub}/trainY.pkl', 'wb') as f:
    pickle.dump(trainY, f)
f.close()
with open(f'{ddir}/{ssub}/pulR.pkl', 'wb') as f:
    pickle.dump(pulR, f)
f.close()

print('Doing subject 5')

subject = '005'
path_dir = f'../../../../../../../project_rppg/Codes/subject_{subject}/trial_001/'
ppgtotal =  pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_005/trial_001/empatica_e4/BVP.csv')
EventMark = pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_005/trial_001/empatica_e4/tags.csv')
dataPath = os.path.join(f'{path_dir}video', '*.MOV')
files = glob.glob(dataPath) 
list.sort(files)

data = []
im_size = (100,100)
cap = cv2.VideoCapture(files[0])
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray  = gray[:,:,1]
    gray =  gray[100:1000, 600:1400]
    gray = cv2.resize(gray, im_size)
    data.append(gray)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
cv2.destroyAllWindows()

data = np.array(data)
evmarknp =  EventMark.to_numpy()
ppgnp =  ppgtotal.to_numpy()
start_gap =  evmarknp[0] -   1599690864
end_point =  evmarknp[1] - evmarknp[0]
ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]
data_align = data[194 : 194 +np.int(end_point*30)+5] 

random.seed(1)
rv = np.arange(0,5000, 1)+1000
np.random.shuffle(rv)
rv =  np.array(rv)
pulR = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1])

trainX = []
trainY = []
data_align = data_align[:,:,:,np.newaxis]
frame_cons = 40 # how many frame to consider at a time

for j, i in enumerate(rv):
    
    img = np.reshape(data_align[i:i+frame_cons,:,:,0], [frame_cons, *im_size])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    p_point = np.int(np.round(i*64/30))
    ppg = pulR[p_point: p_point+85, 0]
    trainY.append(ppg)


ddir = 'replicate_pickle_data'
ssub = 'sub5'
if not os.path.exists(ddir): os.mkdir(ddir)
if not os.path.exists(f'{ddir}/{ssub}'): os.mkdir(f'{ddir}/{ssub}')

with open(f'{ddir}/{ssub}/data_align.pkl', 'wb') as f:
    pickle.dump(data_align, f)
f.close()
with open(f'{ddir}/{ssub}/ppgnp_align.pkl', 'wb') as f:
    pickle.dump(ppgnp_align, f)
f.close()
with open(f'{ddir}/{ssub}/trainX.pkl', 'wb') as f:
    pickle.dump(trainX, f)
f.close()
with open(f'{ddir}/{ssub}/trainY.pkl', 'wb') as f:
    pickle.dump(trainY, f)
f.close()
with open(f'{ddir}/{ssub}/pulR.pkl', 'wb') as f:
    pickle.dump(pulR, f)
f.close()

print('Doing subject 6')

subject = '006'
path_dir = f'../../../../../../../project_rppg/Codes/subject_{subject}/trial_001/'
ppgtotal =  pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_006/trial_001/empatica_e4/BVP.csv')
EventMark = pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_006/trial_001/empatica_e4/tags.csv')
dataPath = os.path.join(f'{path_dir}video', '*.MOV')
files = glob.glob(dataPath) 
list.sort(files)

data = []
im_size = (100,100)
cap = cv2.VideoCapture(files[0])
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray  = gray[:,:,1]
    gray =  gray[0:1000, 400:1400]
    gray = cv2.resize(gray, im_size)
    data.append(gray)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
cv2.destroyAllWindows()

data = np.array(data)
evmarknp =  EventMark.to_numpy()
ppgnp =  ppgtotal.to_numpy()
start_gap =  evmarknp[0] -   1599770587
end_point =  evmarknp[1] - evmarknp[0]
ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]
data_align = data[676 : 676 +np.int(end_point*30)+5]  

random.seed(1)
rv = np.arange(0,5000, 1)+1000
np.random.shuffle(rv)
rv =  np.array(rv)
pulR = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1])

trainX = []
trainY = []
data_align = data_align[:,:,:,np.newaxis]
frame_cons = 40 # how many frame to consider at a time

for j, i in enumerate(rv):
    
    img = np.reshape(data_align[i:i+frame_cons,:,:,0], [frame_cons, *im_size])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    p_point = np.int(np.round(i*64/30))
    ppg = pulR[p_point: p_point+85, 0]
    trainY.append(ppg)


ddir = 'replicate_pickle_data'
ssub = 'sub6'
if not os.path.exists(ddir): os.mkdir(ddir)
if not os.path.exists(f'{ddir}/{ssub}'): os.mkdir(f'{ddir}/{ssub}')

with open(f'{ddir}/{ssub}/data_align.pkl', 'wb') as f:
    pickle.dump(data_align, f)
f.close()
with open(f'{ddir}/{ssub}/ppgnp_align.pkl', 'wb') as f:
    pickle.dump(ppgnp_align, f)
f.close()
with open(f'{ddir}/{ssub}/trainX.pkl', 'wb') as f:
    pickle.dump(trainX, f)
f.close()
with open(f'{ddir}/{ssub}/trainY.pkl', 'wb') as f:
    pickle.dump(trainY, f)
f.close()
with open(f'{ddir}/{ssub}/pulR.pkl', 'wb') as f:
    pickle.dump(pulR, f)
f.close()


print('Doing subject 7')

subject = '007'
path_dir = f'../../../../../../../project_rppg/Codes/subject_{subject}/trial_001/'
ppgtotal =  pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_007/trial_001/empatica_e4/BVP.csv')
EventMark = pd.read_csv('/Users/samuelhmorton/indiv_projects/school/masters/project_rppg/Codes/subject_007/trial_001/empatica_e4/tags.csv')
dataPath = os.path.join(f'{path_dir}video', '*.MOV')
files = glob.glob(dataPath) 
list.sort(files)

data = []
im_size = (100,100)
cap = cv2.VideoCapture(files[0])
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray  = gray[:,:,1]
    gray =  gray[0:1000, 400:1400]
    gray = cv2.resize(gray, im_size)
    data.append(gray)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
cv2.destroyAllWindows()

data = np.array(data)
evmarknp =  EventMark.to_numpy()
ppgnp =  ppgtotal.to_numpy()
start_gap =  evmarknp[0] -   1599686262
end_point =  evmarknp[1] - evmarknp[0]
ppgnp_align =  ppgnp[np.int(start_gap*64):np.int((start_gap+end_point)*64)]
data_align = data[613 : 613 +np.int(end_point*30)+5]  

random.seed(1)
rv = np.arange(0,5000, 1)+1000
np.random.shuffle(rv)
rv =  np.array(rv)
pulR = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1])

trainX = []
trainY = []
data_align = data_align[:,:,:,np.newaxis]
frame_cons = 40 # how many frame to consider at a time

for j, i in enumerate(rv):
    
    img = np.reshape(data_align[i:i+frame_cons,:,:,0], [frame_cons, *im_size])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    p_point = np.int(np.round(i*64/30))
    ppg = pulR[p_point: p_point+85, 0]
    trainY.append(ppg)


ddir = 'replicate_pickle_data'
ssub = 'sub7'
if not os.path.exists(ddir): os.mkdir(ddir)
if not os.path.exists(f'{ddir}/{ssub}'): os.mkdir(f'{ddir}/{ssub}')

with open(f'{ddir}/{ssub}/data_align.pkl', 'wb') as f:
    pickle.dump(data_align, f)
f.close()
with open(f'{ddir}/{ssub}/ppgnp_align.pkl', 'wb') as f:
    pickle.dump(ppgnp_align, f)
f.close()
with open(f'{ddir}/{ssub}/trainX.pkl', 'wb') as f:
    pickle.dump(trainX, f)
f.close()
with open(f'{ddir}/{ssub}/trainY.pkl', 'wb') as f:
    pickle.dump(trainY, f)
f.close()
with open(f'{ddir}/{ssub}/pulR.pkl', 'wb') as f:
    pickle.dump(pulR, f)
f.close()