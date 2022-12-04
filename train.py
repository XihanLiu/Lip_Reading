#%%
#importing libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import natsort
#for reading and displaying images
# from skimage.io import imread
import matplotlib.pyplot as plt
from PIL import Image
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
from dataset import LipDataset
from model import CNN,RNN
from torchvision import transforms
convert_tensor = transforms.ToTensor()
#%%
def constuct_Dataset_withSplitingRatio(PathRoot, data_dirc, Dataset_type, spliting_ratio): 
    label_string_list = []
    TrainData_list = [] # DataALL final (#of all dataset, 50, 100, 29, 3)
    TestData_list = []
    frame_list = []
    for i, label_id in enumerate(os.listdir(PathRoot)):
        Label_root = PathRoot+"/"+label_id
        # constructing the label vector
        num_videos = len(os.listdir(Label_root))
        num_train = int(num_videos*0.8)
        num_test = num_videos - num_train
        if i == 0:
            train_label = np.ones((num_train,1))*i
            test_label = np.ones((num_test,1))*i
        else: 
            train_label = np.concatenate((train_label, np.ones((num_train,1))*i),axis=0)
            test_label = np.concatenate((test_label, np.ones((num_test,1))*i),axis=0)
        label_string_list.append(label_id)
        for j, video_id in tqdm(enumerate(os.listdir(Label_root))):
            if ".mp4" in video_id and video_id[0] != ".":
                Data_Frames = natsort.natsorted(os.listdir(Label_root+'/'+video_id+'/'+data_dirc))
                for Data_Frame in Data_Frames:
                    Data_current = Image.open(os.path.join(Label_root+'/'+video_id+'/'+data_dirc,Data_Frame))
                    Data_current_Tensor = convert_tensor(Data_current)
                    Data_current.close()
                    frame_list.append(Data_current_Tensor)
               
                frame_stack = torch.stack(frame_list)
                print(frame_stack.shape)
                if j < num_train:
                    TrainData_list.append(frame_list)
                    # TrainData_list.append(torch.Tensor(frame_list))
                else: 
                    TestData_list.append(frame_list)
                    # TestData_list.append(frame_list)
            else: 
                 os.remove(Label_root+'/'+video_id)
    
    return TrainData_list, TestData_list, train_label[:,0], test_label[:,0]
    
    
#%% test field
# PATH_ROOT = "D:/Study/Master/Semaster_1/extracted/TRAIN/Lip_3dMatrix/train" # path to the 3d_matrix root
PATH_ROOT = './Data/Lip_frameByFrame'
DatasetSize = "standard"
DatasetType = "III"
# if type I, II, III
if len(DatasetSize) != 0:
    Data_dirc = DatasetSize+"/type"+DatasetType
    # print(Data_dirc)
# if type IV
else: 
    Data_dirc = DatasetType
    # print(Data_dirc)
X_train, X_test, targets_train, targets_test = constuct_Dataset_withSplitingRatio(PATH_ROOT, Data_dirc, DatasetType, 0.8)
X_train = np.asarray(X_train)
# torch.Tensor(X_train)

print(X_train.shape)
#%%
#convert all the variables to pytorch tensor format
#X_train should have shape (num_of_dataset,50,100,29,3) and targets_train(num_of_dataset,1)
#X_test should have shape (num_of_test_dataset,50,100,29,3) and targets_test(num_of_test_dataset,1)
train_x=torch.from_numpy(X_train).float()#change the array into Tensor
del X_train
train_y=torch.from_numpy(targets_train).long()
del targets_train
test_x=torch.from_numpy(X_test).float()
del X_test
test_y=torch.from_numpy(targets_test).long()
del targets_test

#%%
batch_size=10#the batch_size we will use for the training

#pytorch train and test sets
train=torch.utils.data.TensorDataset(train_x,train_y)
del train_x, train_y,
test=torch.utils.data.TensorDataset(test_x,test_y)
del test_x, test_y,

#data loader
train_loader=torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=False)
del train
test_loader=torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=False)#initialize the dataset and divide them into groups
del test


# dataset = LipDataset(train_dir)
# numClasses = dataset.size()

# CNN_Model = CNN()
# RNN_Model = RNN()
# for group in range(numClasses):
#     names,label,length = dataset.getnames(group)
#     print("Load in CNN model...")
    
#     for frame in range(length):
#         train_tensor,train_label = dataset.getitems(group,frame)
#         CNN_Features = CNN_Model(train_tensor)
        
        #print(train_tensor.shape,train_label) # Check output: "torch.Size([3, 50, 100]) AROUND"