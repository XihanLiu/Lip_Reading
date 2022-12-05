# -*- coding: utf-8 -*-

"""
Created on Fri Dec  2 13:18:52 2022

@author: wsycx
"""
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
from scipy.io import savemat

#%%
PATH_ROOT = 'D:/Study/Master/Semaster_1/formated_dataset/TRAIN' # path to the 3d_matrix root
data_type = "typeIV"
data_size = ""

def loadConstructedDataset(PATH):
    data_loaded_dict = loadmat(PATH)
    data_loaded_keys = list(data_loaded_dict.keys())[3]
    data_loaded = data_loaded_dict[data_loaded_keys]
    return data_loaded

def RGB2GRAY(RGBMatrix):
    num_of_dataset, channels, num_rows, num_cols, frames = RGBMatrix.shape
    Gray_matrix = np.ones((num_of_dataset, frames, num_rows, num_cols))
    for i in tqdm(range(num_of_dataset)):
        for f in range(frames):
            RGB_Image = ToCV2ImageShape(RGBMatrix[i,:,:,:,f])
            Gray_matrix[i,f,:,:] = cv2.cvtColor(RGB_Image, cv2.COLOR_RGB2GRAY)
    return Gray_matrix
    
def ToCV2ImageShape(PytorchImage):
    num_channels, num_row, num_col = PytorchImage.shape
    CV2Image = np.ones((num_row, num_col,num_channels)).astype('uint8')
    for i in range(num_channels):
        CV2Image[:,:,i] = PytorchImage[i,:,:]
    return CV2Image

    
def ToPytorchImageShape(CV2Image):
    num_row, num_col, num_channels = CV2Image.shape
    PytorchImage = np.ones((num_channels, num_row, num_col)).astype('uint8')
    for i in range(num_channels):
        PytorchImage[i,:,:] = CV2Image[:,:,i] 
    return PytorchImage
        
def normalization(X):
    '''
    X (#of all dataset, 50, 100, 29)
    '''
    num_data, num_row, num_col, num_frame = X.shape
    X_out = np.ones((num_data, num_frame, num_row, num_col))
    for i in tqdm(range(X.shape[0])):
        for j in range(X.shape[3]):
            current_image = X[i,:,:,j]
            # num_row, num_column = current_image.shape
            pixel_max = np.max(current_image)
            normed_image = current_image[:,:]/pixel_max
            X_out[i,j,:,:] = normed_image
    return X_out
#%% test field
X_train = loadConstructedDataset(PATH_ROOT+"/Lip_3d_"+data_size+data_type+"_X_train")
# X_train = RGB2GRAY(X_train)
X_test = loadConstructedDataset(PATH_ROOT+"/Lip_3d_"+data_size+data_type+"_X_test")
# X_test = RGB2GRAY(X_test)
targets_test = loadConstructedDataset(PATH_ROOT+"/Lip_3d_"+data_size+data_type+"_targets_test").flatten()
targets_train = loadConstructedDataset(PATH_ROOT+"/Lip_3d_"+data_size+data_type+"_targets_train").flatten()
# X_train, X_test, targets_train, targets_test, label_string_list = constuct_Dataset_withSplitingRatio(PATH_ROOT, Data_dirc, DatasetType, 0.9)
#%%
X_train = normalization(X_train)
X_test = normalization(X_test)
#%%
# temp = X_train[1,16,:,:]
# plt.imshow(temp)
# temp_gray = cv2.cvtColor(temp.reshape(25,50,3), cv2.COLOR_RGB2GRAY)

#%%
#convert all the variables to pytorch tensor format
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
del train_x, train_y
test=torch.utils.data.TensorDataset(test_x,test_y)
del test_x, test_y

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
# CNNModel (best 63%)
class CNNModel(nn.Module):
  def __init__(self):
    super(CNNModel,self).__init__()
    self.conv_layer1=self._conv_layer_set(29,128)
    self.conv_layer2=self._conv_layer_set(128,64)
    self.conv_layer3=self._conv_layer_set(64,16)
    #4,10,1
    # self.fc1=nn.Linear(50176, 2048)
    # self.fc2 = nn.Linear(2048, 64)
    self.fc1=nn.Linear(49280,num_classes)
    self.relu = nn.LeakyReLU()
    self.sigmoid = nn.Sigmoid()
    self.batch=nn.BatchNorm1d(49280)
    self.drop=nn.Dropout(p=0.12)        
        
  def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride = 1, padding=0),
        nn.LeakyReLU(),
        nn.MaxPool2d((3, 3),stride=1),
        )
        return conv_layer
    

  def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.conv_layer3(out)
        out = out.view(out.size(0), -1)
        out = self.batch(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop(out)
        return out

class CNNModel_2(nn.Module):
    def __init__(self):
      super(CNNModel_2,self).__init__()
      self.conv_layer1=self._conv_layer_set(29,1028)
      self.conv_layer2=self._conv_layer_set_1(1028,256)
      self.conv_layer3=self._conv_layer_set_1(256,64)
      #4,10,1
      # self.fc1=nn.Linear(50176, 2048)
      # self.fc2 = nn.Linear(2048, 64)
      self.fc1=nn.Linear(34944,num_classes)
      self.relu = nn.LeakyReLU()
      self.sigmoid = nn.Sigmoid()
      self.batch=nn.BatchNorm1d(34944)
      self.drop=nn.Dropout(p=0.12)        
          
    def _conv_layer_set(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv2d(in_c, out_c, kernel_size=(10, 10), stride = 1, padding=0),
          nn.LeakyReLU(),
          nn.MaxPool2d((3, 3),stride=1),
          )
          return conv_layer
    
    def _conv_layer_set_1(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride = 1, padding=0),
        nn.LeakyReLU(),
        nn.MaxPool2d((2, 2),stride=1),    
        )
        return conv_layer
    
    def forward(self, x):
          # Set 1
          out = self.conv_layer1(x)
          out = self.conv_layer2(out)
          out = self.relu(out)
          # print(out.shape)
          out = self.conv_layer3(out)
          out = out.view(out.size(0), -1)
          out = self.batch(out)
          out = self.fc1(out)
          out = self.relu(out)
          out = self.drop(out)
          return out
      
class CNNModel_3(nn.Module):
    def __init__(self):
      super(CNNModel_3,self).__init__()
      self.conv_layer1=self._conv_layer_set(29,29)
      self.conv_layer2=self._conv_layer_set_1(29,29)
      self.conv_layer3=self._conv_layer_set_1(29,29)
      #4,10,1
      # self.fc1=nn.Linear(50176, 2048)
      # self.fc2 = nn.Linear(2048, 64)
      self.fc1=nn.Linear(52374,num_classes)
      self.relu = nn.LeakyReLU()
      self.sigmoid = nn.Sigmoid()
      self.batch=nn.BatchNorm1d(52374)
      self.drop=nn.Dropout(p=0.12)        
          
    def _conv_layer_set(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv2d(in_c, out_c, kernel_size=(10, 10), stride = 1, padding=0),
          nn.LeakyReLU(),
          nn.MaxPool2d((5, 5),stride=1),
          )
          return conv_layer
    
    def _conv_layer_set_1(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(5, 5), stride = 1, padding=0),
        nn.LeakyReLU(),
        nn.MaxPool2d((3, 3),stride=1),    
        )
        return conv_layer
    
    def forward(self, x):
          # Set 1
          out = self.conv_layer1(x)
          out = self.conv_layer2(out)
          out = self.relu(out)
          # print(out.shape)
          out = self.conv_layer3(out)
          out = out.view(out.size(0), -1)
          out = self.batch(out)
          out = self.fc1(out)
          out = self.relu(out)
          out = self.drop(out)
          return out
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#%%

#Definition of hyperparameters
n_iters = 2000
num_epochs = 20


# Create CNN
model = CNNModel_3().to(device)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.00002
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
        images, labels = images.to(device), labels.to(device)
        batch_size = images.shape[0]
        if batch_size == 10:
            labels = Variable(labels)
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(images)
            # Calculate softmax and ross entropy loss
            loss = error(outputs, labels)
            # print(loss.shape)
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            count += 1
            
       
    # Calculate Accuracy         
    correct = 0
    total = 0
    with torch.no_grad():
        # Iterate through test dataset
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]
            if batch_size == 10: 
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
    # # Print Loss
    # print('iteration: {}  Loss: {}  Accuracy: {} %'.format(i, loss.data, accuracy))
    # # Print Loss
    # print("--------------------------------------------------------")
    print('Epoch: {}  Loss: {}  Accuracy: {} %'.format(epoch, loss.data, accuracy))



#%%
