# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 22:56:54 2022

@author: wsycx
"""

"""
Created on Fri Dec  2 13:18:52 2022

@author: wsycx
"""

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


#%%
PATH_ROOT = "D:/Study/Master/Semaster_1/extracted/TRAIN/Lip_frameByFrame" # path to the 3d_matrix root
DatasetSize = "small"
DatasetType = "III"
# if type I, II, III
if len(DatasetSize) != 0:
    Data_dirc = DatasetSize+"/type"+DatasetType
# if type IV
else: 
    Data_dric = DatasetType


def constuct_Dataset_withSplitingRatio(PathRoot, data_dirc, Dataset_type, spliting_ratio): 
    # train_label_list = []
    # test_label_list = []
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
                for m, image_id in enumerate(os.listdir(image_folder_dirc)):
                    if ".jpg" in image_id and image_id[0] != ".":
                        current_image = plt.imread(image_folder_dirc+'/'+image_id)
                        if m == 0:
                            single_image_num_rows, single_image_num_cols, single_image_num_chennels =  current_image.shape
                            same_label_image = np.ones((single_image_num_chennels, single_image_num_rows, single_image_num_cols,num_frames)).astype('uint8')
                        same_label_image[:,:,:,m] = current_image.reshape(single_image_num_chennels, single_image_num_rows, single_image_num_cols)
                    else:
                        os.remove(Label_root+'/'+video_id)
                if j < num_train:
                    TrainData_list.append(same_label_image)
                else: 
                    TestData_list.append(same_label_image)
            else: 
                 os.remove(Label_root+'/'+video_id)
    return np.array(TrainData_list).astype("uint8"), np.array(TestData_list).astype("uint8"), train_label[:,0], test_label[:,0], label_string_list
    
def normalization(X):
    '''
    X (#of all dataset, 3, 50, 100, 29)
    '''
    for i in tqdm(range(X.shape[0])):
        for j in range(X.shape[4]):
            current_image = X[i,:,:,:,j]
            num_channel, num_row, num_column = current_image.shape
            current_image = current_image.flatten()
            normed_image = current_image/norm(current_image)
            X[i,:,:,:,j] = normed_image.reshape(num_channel, num_row, num_column)
    return X
#%% test field
X_train, X_test, targets_train, targets_test, label_string_list = constuct_Dataset_withSplitingRatio(PATH_ROOT, Data_dirc, DatasetType, 0.9)
print(X_train.shape)