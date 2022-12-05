#%%
# 3d-cnn implement
#importing libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os 
import cv2

#for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.io import loadmat

#for creating validation set
from sklearn.model_selection import train_test_split
#for evaluating the model
from sklearn.metrics import accuracy_score

#Pytorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py
import shutil
#from plot3D import *

from numpy.linalg import norm
import scipy.io
#%%
PATH_ROOT = '/Users/liuxihan/Desktop/Lip_Reading/Data/Lip_frameByFrame' # path to the 3d_matrix root
DatasetSize = ""
DatasetType = "typeIV"
# if type I, II, III
if len(DatasetSize) != 0:
    Data_dirc = DatasetSize+"/type"+DatasetType
# if type IV
else: 
    Data_dric = DatasetType

def constuct_Dataset_withSplitingRatio(PathRoot, data_dirc, Dataset_type, spliting_ratio): 
    label_string_list = []
    TrainData_list = [] # DataALL final (#of all dataset, 3, 50, 100, 29)
    TestData_list = []
    for i, label_id in enumerate(os.listdir(PathRoot)):
        Label_root = PathRoot+"/"+label_id
        # constructing the label vector
        num_videos = len(os.listdir(Label_root))
        num_train = int(num_videos*spliting_ratio)
        num_test = num_videos - num_train
        if i == 0:
            train_label = (np.ones((num_train,1))*i).astype('uint8')
            test_label = (np.ones((num_test,1))*i).astype('uint8')
        else: 
            train_label = np.concatenate((train_label, np.ones((num_train,1))*i),axis=0)
            test_label = np.concatenate((test_label, np.ones((num_test,1))*i),axis=0)
        label_string_list.append(label_id)
        for j, video_id in tqdm(enumerate(os.listdir(Label_root))):
            if ".mp4" in video_id and video_id[0] != ".":
                image_folder_dirc = Label_root+'/'+video_id+'/'+data_dirc
                num_frames = len(os.listdir(image_folder_dirc))
                if num_frames !=29:
                    print(num_frames,video_id)
                if num_frames == 29:
                    for m, image_id in enumerate(os.listdir(image_folder_dirc)):
                        if ".mat" in image_id and image_id[0] != ".":
                            current_image = scipy.io.loadmat(image_folder_dirc+'/'+image_id)
                            current_image = current_image['dist_array']
                            # current_image = plt.imread(image_folder_dirc+'/'+image_id)
                            if m == 0:
                                single_image_num_rows, single_image_num_cols =  current_image.shape
                                same_label_image = np.ones((single_image_num_rows, single_image_num_cols,num_frames)).astype('uint8')
                            same_label_image[:,:,m] = current_image
                            
                        else:
                            os.remove(Label_root+'/'+video_id)
                    if j < num_train:
                        TrainData_list.append(same_label_image)
                    else: 
                        TestData_list.append(same_label_image)
            else: 
                os.remove(Label_root+'/'+video_id)
        
    TrainData_array = np.array(TrainData_list)
    TestData_array = np.array(TestData_list)
    

    return TrainData_array, TestData_array, train_label[:,0], test_label[:,0], label_string_list

X_train, X_test, targets_train, targets_test, label_string_list = constuct_Dataset_withSplitingRatio(PATH_ROOT, Data_dric, DatasetType, 0.9)
#%% save file
mdic_X_train = {"Lip_3d_" + DatasetType +"_X_train": X_train}
scipy.io.savemat("Lip_3d_"+ DatasetType+"_X_train.mat",mdic_X_train)

mdic_X_test = {"Lip_3d_" + DatasetType +"_X_test": X_test}
scipy.io.savemat("Lip_3d_"+ DatasetType+"_X_test.mat",mdic_X_test)

mdic_targets_train = {"Lip_3d_"+ DatasetType +"_targets_train": targets_train}
scipy.io.savemat("Lip_3d_"+ DatasetType+"_targets_train.mat",mdic_targets_train)

mdic_targets_test = {"Lip_3d_"+ DatasetType +"_targets_test": targets_test}
scipy.io.savemat("Lip_3d_"+DatasetType+"_targets_test.mat",mdic_targets_test)
