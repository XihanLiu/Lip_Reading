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

#Pytorch libraries and modulesw
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
# PATH_ROOT = '/Users/mikewang/Library/CloudStorage/OneDrive-JohnsHopkins/Study/Master/Semaster_1/EN.520.612/Lip_Reading/formated_dataset/TRAIN/small/typeII' # path to the 3d_matrix root
# data_type = "_II"
# data_size = "small"

PATH_ROOT = 'F:/newdata' # path to the 3d_matrix root
data_type = "_I"
data_size = "small"
Trained_model_Root = "Trained_model"


def loadConstructedDataset(PATH):
    data_loaded_dict = loadmat(PATH)
    data_loaded_keys = list(data_loaded_dict.keys())[3]
    data_loaded = data_loaded_dict[data_loaded_keys]
    return data_loaded
    
def normalization(X):
    '''
    X (#of all dataset, 3, 50, 100, 29)
    '''
    shape_of_X = X.shape
    X_out = np.ones((shape_of_X))
    for i in tqdm(range(X.shape[0])):
        for j in range(X.shape[4]):
            current_image = X[i,:,:,:,j]
            num_channel, num_row, num_column = current_image.shape
            
            for c in range(num_channel):
                # current_image = current_image.flatten()
                pixel_max = np.max(current_image[c,:,:])
                normed_image = current_image[c,:,:]/pixel_max
                X_out[i,c,:,:,j] = normed_image
    return X_out
#%% test field
X_train = loadConstructedDataset(PATH_ROOT+"/Lip_frameByFrame3d_"+data_size+data_type+"_X_train")
X_test = loadConstructedDataset(PATH_ROOT+"/Lip_frameByFrame3d_"+data_size+data_type+"_X_test")
targets_test = loadConstructedDataset(PATH_ROOT+"/Lip_frameByFrame3d_"+data_size+data_type+"_targets_test").flatten()
targets_train = loadConstructedDataset(PATH_ROOT+"/Lip_frameByFrame3d_"+data_size+data_type+"_targets_train").flatten()
# X_train, X_test, targets_train, targets_test, label_string_list = constuct_Dataset_withSplitingRatio(PATH_ROOT, Data_dirc, DatasetType, 0.9)
#%%
X_train = normalization(X_train)
X_test = normalization(X_test)
#%%
# temp = X_train[1,:,:,:,16]
# temp2 = np.ones((50,100,3))

# for i in range(temp.shape[0]):
#     temp2[:,:,i] = temp[i,:,:]
#     plt.imshow(temp[i,:,:])
#     plt.show()
# plt.imshow(temp2.astype('uint8'))
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
batch_size=32#the batch_size we will use for the training

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

    self.conv_layer1=self._conv_layer_set(3,16)
    #50-2=48
    #100-2=98
    #29-2=27
    #24，49，13
    self.conv_layer2=self._conv_layer_set(16,32)
    #24-2=22
    #49-2=47
    #13-2=11
    #11，23，5
    self.conv_layer3=self._conv_layer_set(32,16)
    #4,10,1
    self.fc1=nn.Linear(50176, 2048)
    self.fc2 = nn.Linear(2048, 64)
    self.fc3=nn.Linear(2048,num_classes)
    self.relu = nn.LeakyReLU()
    self.sigmoid = nn.Sigmoid()
    self.batch=nn.BatchNorm1d(50176)
    self.drop=nn.Dropout(p=0.12)        
        
  def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 2), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((3, 3, 1)),
        )
        return conv_layer
    

  def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        # out = self.conv_layer2(out)
        out = self.relu(out)
        # print(out.shape)
        # out = self.conv_layer3(out)
        out = out.view(out.size(0), -1)
        out = self.batch(out)
        out = self.fc1(out)
        out = self.relu(out)
        
        out = self.drop(out)
        # out = self.fc2(out)
        out = self.fc3(out)
        out = self.relu(out)
        
        return out


# max 64%
class CNNModel_2(nn.Module):
    def __init__(self):
      super(CNNModel_2,self).__init__()
    
      self.conv_layer1=self._conv_layer_set(3,128)
      #50-2=48
      #100-2=98
      #29-2=27
      #24，49，13
      self.conv_layer2=self._conv_layer_set_2(128,32)
      #24-2=22
      #49-2=47
      #13-2=11
      #11，23，5
      self.conv_layer3=self._conv_layer_set(32,16)
      #4,10,1
      self.fc1=nn.Linear(3584, 2048)
      self.fc2 = nn.Linear(2048, 64)
      self.fc3=nn.Linear(3584,num_classes)
      self.relu = nn.LeakyReLU()
      self.sigmoid = nn.Sigmoid()
      self.batch=nn.BatchNorm1d(3584)
      self.drop=nn.Dropout(p=0.12)        
          
    def _conv_layer_set(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 2), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((3, 3, 1)),
          )
          return conv_layer
    
    def _conv_layer_set_2(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 1), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((3, 3, 1)),
          )
          return conv_layer
    

    def forward(self, x):
          # Set 1
          out = self.conv_layer1(x)
          out = self.conv_layer2(out)
          out = self.relu(out)
          # print(out.shape)
          # out = self.conv_layer3(out)
          out = out.view(out.size(0), -1)
          out = self.batch(out)
          # out = self.fc1(out)
          # out = self.relu(out)
          
          out = self.drop(out)
          # out = self.fc2(out)
          out = self.fc3(out)
          out = self.relu(out)
          
          return out

# max 68%
class CNNModel_3(nn.Module):
    def __init__(self):
      super(CNNModel_3,self).__init__()
    
      self.conv_layer1=self._conv_layer_set(3,256)
      self.conv_layer2=self._conv_layer_set_2(256,128)
      self.conv_layer3=self._conv_layer_set_3(128,16)
      self.fc1=nn.Linear(14336, 2048)
      self.fc2 = nn.Linear(2048, 64)
      self.fc3=nn.Linear(14336,num_classes)
      self.relu = nn.LeakyReLU()
      self.sigmoid = nn.Sigmoid()
      self.batch=nn.BatchNorm1d(14336)
      self.drop=nn.Dropout(p=0.12)        
          
    def _conv_layer_set(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 2), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((3, 3, 1)),
          )
          return conv_layer
    
    def _conv_layer_set_2(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 1), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((3, 3, 1)),
          )
          return conv_layer
    
    def _conv_layer_set_3(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(2, 2, 1), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((2, 2, 1)),
          )
          return conv_layer

    def forward(self, x):
          # Set 1
          out = self.conv_layer1(x)
          out = self.conv_layer2(out)
          out = self.relu(out)
          # print(out.shape)
          # out = self.conv_layer3(out)
          out = out.view(out.size(0), -1)
          out = self.batch(out)
          # out = self.fc1(out)
          # out = self.relu(out)
          
          out = self.drop(out)
          # out = self.fc2(out)
          out = self.fc3(out)
          out = self.relu(out)
          
          return out
      
# max 61%
class CNNModel_4(nn.Module):
    def __init__(self):
      super(CNNModel_4,self).__init__()
    
      self.conv_layer1=self._conv_layer_set(3,256)
      self.conv_layer2=self._conv_layer_set_2(256,128)
      self.conv_layer3=self._conv_layer_set_3(128,16)
      # self.conv_layer4=self._conv_layer_set_3(56,16)
      self.fc1=nn.Linear(4992, 2048)
      # self.fc2 = nn.Linear(2048, 64)
      self.fc3=nn.Linear(4992,num_classes)
      self.relu = nn.LeakyReLU()
      self.sigmoid = nn.Sigmoid()
      self.batch=nn.BatchNorm1d(4992)
      self.drop=nn.Dropout(p=0.12)        
          
    def _conv_layer_set(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((3, 3, 1)),
          )
          return conv_layer
    
    def _conv_layer_set_2(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 2), padding=0),
          nn.LeakyReLU(),
          # nn.MaxPool3d((2, 2, 1)),
          )
          return conv_layer
    
    def _conv_layer_set_3(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(2, 2, 1), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((2, 2, 1)),
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
          # out = self.fc1(out)
          # out = self.relu(out)
          
          out = self.drop(out)
          # out = self.fc2(out)
          out = self.fc3(out)
          out = self.relu(out)
          
          return out

# max 68%
class CNNModel_5(nn.Module):
    def __init__(self):
      super(CNNModel_5,self).__init__()
    
      self.conv_layer1=self._conv_layer_set(3,256)
      self.conv_layer2=self._conv_layer_set_2(256,128)
      self.conv_layer3=self._conv_layer_set_3(128,16)
      self.fc1=nn.Linear(6912, 2048)
      # self.fc2 = nn.Linear(2048, 64)
      self.fc3=nn.Linear(2048,num_classes)
      self.relu = nn.LeakyReLU()
      self.sigmoid = nn.Sigmoid()
      self.batch=nn.BatchNorm1d(6912)
      self.drop=nn.Dropout(p=0.12)        
          
    def _conv_layer_set(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 2), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((3, 3, 1)),
          )
          return conv_layer
    
    def _conv_layer_set_2(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 2), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((5, 5, 1)),
          )
          return conv_layer
    
    def _conv_layer_set_3(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(2, 2, 1), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((2, 2, 1)),
          )
          return conv_layer

    def forward(self, x):
          # Set 1
          out = self.conv_layer1(x)
          out = self.conv_layer2(out)
          out = self.relu(out)
          # print(out.shape)
          # out = self.conv_layer3(out)
          out = out.view(out.size(0), -1)
          out = self.batch(out)
          out = self.fc1(out)
          out = self.relu(out)
          
          out = self.drop(out)
          # out = self.fc2(out)
          out = self.fc3(out)
          out = self.relu(out)
          
          return out    

# max 68%
class CNNModel_6(nn.Module):
    def __init__(self):
      super(CNNModel_6,self).__init__()
    
      self.conv_layer1=self._conv_layer_set(3,256)
      self.conv_layer2=self._conv_layer_set_2(256,128)
      self.conv_layer3=self._conv_layer_set_3(128,16)
      self.fc1=nn.Linear(14336, 2048)
      self.fc2 = nn.Linear(2048, 64)
      self.fc3=nn.Linear(64,num_classes)
      self.relu = nn.LeakyReLU()
      self.sigmoid = nn.Sigmoid()
      self.batch=nn.BatchNorm1d(14336)
      self.drop=nn.Dropout(p=0.12)        
          
    def _conv_layer_set(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 2), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((3, 3, 1)),
          )
          return conv_layer
    
    def _conv_layer_set_2(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 1), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((3, 3, 1)),
          )
          return conv_layer
    
    def _conv_layer_set_3(self, in_c, out_c):
          conv_layer = nn.Sequential(
          nn.Conv3d(in_c, out_c, kernel_size=(2, 2, 1), padding=0),
          nn.LeakyReLU(),
          nn.MaxPool3d((2, 2, 1)),
          )
          return conv_layer

    def forward(self, x):
          # Set 1
          out = self.conv_layer1(x)
          out = self.conv_layer2(out)
          out = self.relu(out)
          # print(out.shape)
          # out = self.conv_layer3(out)
          out = out.view(out.size(0), -1)
          out = self.batch(out)
          out = self.fc1(out)
          # out = self.relu(out)
          
          out = self.drop(out)
          out = self.fc2(out)
          out = self.fc3(out)
          out = self.relu(out)
          
          return out

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#%%

#Definition of hyperparameters
n_iters = 2000
num_epochs = 40


# Create CNN
model = CNNModel_6().to(device)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.00005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%
#train model
# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
val_loss_list=[]
train_acc_list=[]
max_acc = 0
for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0 
    for i, (images, labels) in enumerate(train_loader):
        # print(images.shape)
        images, labels = images.to(device), labels.to(device)
        batch_size = images.shape[0]
        if batch_size == 32:
            labels = Variable(labels)
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(images)
            # Calculate softmax and ross entropy loss
            predicted = torch.max(outputs, 1)[1]
            loss = error(outputs, labels)
            train_loss = loss.data
            train_total += len(labels)
            train_correct += (predicted == labels).sum()
            
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            count += 1
            
       
    # Calculate Accuracy         
    correct = 0
    total = 0
    count_labels=np.zeros(11)
    count_labels_pre=np.zeros(11)
    model.eval()
    with torch.no_grad():
        # Iterate through test dataset
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]
            if batch_size == 32: 
                # Forward propagation
                outputs = model(images)
                # print(outputs)
                # Get predictions from the maximum value
                predicted = torch.max(outputs, 1)[1]
                # Total number of labels
                total += len(labels)
                correct += (predicted == labels).sum()
                val_loss = error(outputs, labels)
            if epoch == 39:
                for ii in range(10):
                    count_labels[labels[ii]]+=1
                    if predicted[ii]==labels[ii]:
                        count_labels_pre[predicted[ii]]+=1
    accuracy = 100 * correct / float(total)
    train_acc = (train_correct/train_total)*100
    # store loss and iteration
    loss_list.append(loss.data)
    val_loss_list.append(val_loss.data)
    iteration_list.append(count)
    train_acc_list.append(train_acc)
    accuracy_list.append(accuracy)
    if accuracy > max_acc:
        max_acc = accuracy
        best_model = model
        print("---- NEW BEST")
    print('Epoch:{}  Train_Loss:{} Train_Acc :{}% Val_Loss:{}  Val_Acc:{}%'.format(epoch,train_loss, train_acc, val_loss.data, accuracy))


torch.save(best_model, Trained_model_Root + "/"+data_size+data_type+"_rgb_3d.pt")
#%% show the results
epochs = range(len(accuracy_list))
for i in range(len(accuracy_list)):
    accuracy_list[i] = accuracy_list[i].cpu()
for i in range(len(loss_list)):
    loss_list[i] = loss_list[i].cpu()
for i in range(len(train_acc_list)):
    train_acc_list[i] = train_acc_list[i].cpu()
for i in range(len(val_loss_list)):
    val_loss_list[i] = val_loss_list[i].cpu()
plt.plot(epochs,accuracy_list,'b',label='Validation Accuracy')
plt.plot(epochs,train_acc_list,'r',label='Training Accuracy')
# plt.plot(torch.tensor(history["accuracy"],device='cpu'),label='validation accuracy')
plt.xlabel('num_of_epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs. epochs')
plt.legend()
plt.show()
plt.plot(epochs,loss_list,'r',label='Training Loss')
plt.plot(epochs,val_loss_list,'b',label='Validation loss')
plt.xlabel('num_of_epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. epochs')
plt.legend()
plt.show()
#%% accuracy for words
count_labels[4]-=2
count_labels[2]-=1
count_labels[6]-=2
count_labels[0]-=1
count_labels[8]-=1
count_labels[9]-=2
count_labels[5]-=1
count_labels[1]-=2
count_labels[3]-=1
ratio=count_labels_pre/count_labels
name_list=['AHEAD','AROUND','AGAIN','ACTUALLY','ANSWER','ALLOWED','ALLOW','ACCESS','ALWAYS','AMERICA','ABSOLUTELY']
f, ax = plt.subplots(figsize=(18,5)) # set the size that you'd like (width, height)\
plt.bar(range(len(ratio)),ratio,tick_label=name_list)
plt.title('Lip reading accuracy using 3d-cnn')
plt.ylabel('accuracy')
ax.legend(fontsize = 1)
# plt.plot(figsize=(10,20))
# # 
# plt.plot(figsize=(10,20))
# plt.show()

