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


#%%
PATH_ROOT = "D:/Study/Master/Semaster_1/extracted/TRAIN/Lip_3dMatrix/train" # path to the 3d_matrix root
DatasetSize = "standard"
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
    TrainData_list = [] # DataALL final (#of all dataset, 50, 100, 29, 3)
    TestData_list = []
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
                Data_current = loadmat(Label_root+'/'+video_id+'/'+data_dirc+"/_3dType"+Dataset_type)
                Data_current_keys = list(Data_current.keys())
                Data_current = Data_current[Data_current_keys[3]]
                num_frames, num_rows, num_coloumns, num_channels = Data_current.shape
                Data_current = Data_current.reshape(num_rows, num_coloumns,num_frames,num_channels)
                if j < num_train:
                    TrainData_list.append(Data_current)
                else: 
                    TestData_list.append(Data_current)
            else: 
                 os.remove(Label_root+'/'+video_id)
    return np.array(TrainData_list), np.array(TestData_list), train_label[:,0], test_label[:,0]
    
    
#%% test field
X_train, X_test, targets_train, targets_test = constuct_Dataset_withSplitingRatio(PATH_ROOT, Data_dirc, DatasetType, 0.8)


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
    self.fc1=nn.Linear(11*23*5*64, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3=nn.Linear(64,num_classes)
    self.relu = nn.LeakyReLU()
    self.batch=nn.BatchNorm1d(128)
    self.drop=nn.Dropout(p=0.15)        
        
  def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
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
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%%
#train model
# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape)
        train = Variable(images.view(10,3,50,100,29))
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        
        count += 1
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                
                test = Variable(images.view(10,3,50,100,29))
                # Forward propagation
                outputs = model(test)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
