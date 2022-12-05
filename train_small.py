#%%
#importing libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
# import natsort
#for reading and displaying images
# from skimage.io import imread
import matplotlib.pyplot as plt
from PIL import Image
#for creating validation set
# from sklearn.model_selection import train_test_split
#for evaluating the model
# from sklearn.metrics import accuracy_score
#Pytorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from model_small import CNN,RNN
from fit import fit
from scipy.io import loadmat,savemat

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("cpu")
#%%
# mps_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(mps_device)
# mps_device = 'cpu'
#%%

X_train = loadmat('Lip_frames_small_I_X_train.mat')['Lip_frames_smallI_X_train']
X_test = loadmat('Lip_frames_small_I_X_test.mat')['Lip_frames_smallI_X_test']
targets_train = loadmat('Lip_frames_small_I_targets_train.mat')['Lip_frames_smallI_targets_train']
targets_test = loadmat('Lip_frames_small_I_targets_test.mat')['Lip_frames_smallI_targets_test']
#%%
#convert all the variables to pytorch tensor format
#X_train should have shape (num_of_dataset,50,100,29,3) and targets_train(num_of_dataset,1)
#X_test should have shape (num_of_test_dataset,50,100,29,3) and targets_test(num_of_test_dataset,1)
train_x=torch.from_numpy(X_train).float()#change the array into Tensor

# del X_train
train_y=torch.from_numpy(targets_train[0,:]).long()
# del targets_train
test_x=torch.from_numpy(X_test).float()
# del X_test
test_y=torch.from_numpy(targets_test[0,:]).long()
# del targets_test

#%%
batch_size=10#the batch_size we will use for the training

#pytorch train and test sets
train=torch.utils.data.TensorDataset(train_x,train_y)
# del train_x, train_y,
test=torch.utils.data.TensorDataset(test_x,test_y)
# del test_x, test_y,

#data loader
train_loader=torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=True)
# del train
test_loader=torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=True)#initialize the dataset and divide them into groups
# del test
#%%
#Definition of hyperparameters
n_iters = 400
num_epochs = 10
#num_epochs = n_iters / (len(train_x) / batch_size)
#num_epochs = int(num_epochs)

# Create CNN
CNN_model = CNN()
CNN_model.to(mps_device)
RNN_model = RNN()
RNN_model.to(mps_device)
# model.cuda()
# print(CNN_model)
# print(RNN_model)

# Loss Function 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.00001
optimizer = torch.optim.Adam(CNN_model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(RNN_model.parameters(), lr=learning_rate)
#%%
#train model
# CNN model training
# print("Start training...")
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # print(i)
        images,labels = images.to(mps_device),labels.to(mps_device)
        # Clear gradients
        batch_size = images.shape[0]
        if batch_size == 10:
            optimizer.zero_grad()
            Outputs_Tensor = torch.FloatTensor(10,10,29) #(num_classes * frames)
            for j in range(images.shape[4]):
            # Forward propagation
                outputs_CNN = CNN_model(images[:,:,:,:,j]) #shape: (10,10,1)
                Outputs_Tensor[:,:,j] = outputs_CNN 
            Outputs_Tensor = torch.reshape(Outputs_Tensor,(10,-1))
            outputs_RNN = RNN_model(Outputs_Tensor.to(mps_device))
           
            loss = error(outputs_RNN[:,0], labels.float())
            
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()
        
            count += 1
    # print("Start testing...")
    # Calculate Accuracy         
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in test_loader:
        images,labels = images.to(mps_device),labels.to(mps_device)
        batch_size = images.shape[0]
        if batch_size == 10:
            Outputs_Tensor = torch.FloatTensor(10,10,29) #(num_classes * frames)
            for j in range(images.shape[4]):
            # Forward propagation
                outputs_CNN = CNN_model(images[:,:,:,:,j]) #shape: (10,10,1)
                Outputs_Tensor[:,:,j] = outputs_CNN 
    
            Outputs_Tensor = torch.reshape(Outputs_Tensor,(10,-1))
            outputs_RNN = RNN_model(Outputs_Tensor.to(mps_device))
            
            
            
            # Get predictions from the maximum value
            predicted = torch.max(outputs_RNN, 1)[1]
            # print(predicted)
            # Total number of labels
            total += len(labels)
            # print(total)
            # print(labels)
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

