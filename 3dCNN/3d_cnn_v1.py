# -*- coding: utf-8 -*-
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
PATH_ROOT = "D:/Study/Master/Semaster_1/extracted/TRAIN/Lip_3dMatrix/train" # path to the 3d_matrix root
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
                Data_current = loadmat(Label_root+'/'+video_id+'/'+data_dirc+"/_3dType"+Dataset_type)
                Data_current_keys = list(Data_current.keys())
                Data_current = Data_current[Data_current_keys[3]]
                num_frames, num_rows, num_coloumns, num_channels = Data_current.shape
                # Data_current = Data_current.reshape(num_channels, num_rows,num_coloumns,num_frames)
                # Data_current = Data_current.reshape(num_rows*num_coloumns, num_frames,num_channels) # (50*100, 29, 3)
                # Data_current = Data_current/ norm(Data_current,axis=0)
                Data_current_reshaped = np.zeros((num_channels,num_rows,num_coloumns,num_frames)).astype('uint8')
                for m in range(num_frames):
                    for n in range(num_channels):
                        Data_current_reshaped[n,:,:,n] = Data_current[m,:,:,n]
                # Data_current = Data_current.reshape(num_rows,num_coloumns,num_frames,num_channels)
                
                
                if j < num_train:
                    TrainData_list.append(Data_current_reshaped)
                else: 
                    TestData_list.append(Data_current_reshaped)
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

#%%
# X_train = normalization(X_train)
# X_test = normalization(X_test)
# #%%
# temp = X_train[0,:,:,:,0]
# plt.imshow(temp[0,:,:])
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
train_loader=torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=True)
del train
test_loader=torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=True)#initialize the dataset and divide them into groups
del test


#%%
#implement the model
num_classes=11
#50*100*29
#create CNN model
class CNNModel(nn.Module):
  def __init__(self):
    super(CNNModel,self).__init__()

    self.conv_layer1=self._conv_layer_set(3,32)
    #50-2=48
    #100-2=98
    #29-2=27
    #24，49，13
    self.conv_layer2=self._conv_layer_set(32,64)
    #24-2=22
    #49-2=47
    #13-2=11
    #11，23，5
    self.fc1=nn.Linear(11264, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3=nn.Linear(64,num_classes)
    self.relu = nn.LeakyReLU()
    self.batch=nn.BatchNorm1d(128)
    self.drop=nn.Dropout(p=0.15)        
        
  def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 5), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

  def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

#Definition of hyperparameters
n_iters = 2000
num_epochs = 20
#num_epochs = n_iters / (len(train_x) / batch_size)
#num_epochs = int(num_epochs)

# Create CNN
model = CNNModel()
# model.cuda()
print(model)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%
#train model
# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # print(images.shape)
        batch_size = images.shape[0]
        if batch_size == 10:
            # train = Variable(images.view(10,3,50,100,29))
            
            # train = Variable(images)
            labels = Variable(labels)
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(images)
            # Calculate softmax and ross entropy loss
            loss = error(outputs, labels)
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()
        
            count += 1
        if count % 10 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                batch_size = images.shape[0]
                if batch_size == 10: 
                    # test = Variable(images.view(batch_size,3,50,100,29))
                    # test = Variable(images)
                    # print(test.shape)
                    # Forward propagation
                    outputs = model(images)
                    # print(outputs)
    
                    # Get predictions from the maximum value
                    predicted = torch.max(outputs, 1)[1]
                    
                    # Total number of labels
                    total += len(labels)
                    correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            # Print Loss
            print('iteration: {}  Loss: {}  Accuracy: {} %'.format(i, loss.data, accuracy))
    # Print Loss
    print("--------------------------------------------------------")
    print('Epoch: {}  Loss: {}  Accuracy: {} %'.format(epoch, loss.data, accuracy))



#%%
