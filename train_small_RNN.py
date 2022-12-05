#%%
#importing libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from fit import fit
from scipy.io import loadmat,savemat

# Check that MPS is available
mps_device = torch.device("cpu")
#%%
X_train = loadmat('Lip_frames_small_III_X_train.mat')['Lip_frames_smallIII_X_train']
X_test = loadmat('Lip_frames_small_III_X_test.mat')['Lip_frames_smallIII_X_test']
targets_train = loadmat('Lip_frames_small_III_targets_train.mat')['Lip_frames_smallIII_targets_train']
targets_test = loadmat('Lip_frames_small_III_targets_test.mat')['Lip_frames_smallIII_targets_test']
#%%
train_x=torch.from_numpy(X_train).float()#change the array into Tensor
train_y=torch.from_numpy(targets_train[0,:]).long()
test_x=torch.from_numpy(X_test).float()
test_y=torch.from_numpy(targets_test[0,:]).long()
del X_train, X_test, targets_train, targets_test
#%%
batch_size=32#the batch_size we will use for the training
#Definition of hyperparameters
n_iters = 2000
num_epochs = 30
#pytorch train and test sets
train=torch.utils.data.TensorDataset(train_x,train_y)
test=torch.utils.data.TensorDataset(test_x,test_y)
del train_x, train_y, test_x, test_y

#data loader
train_loader=torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=True)#initialize the dataset and divide them into groups
del train, test
#%%
# Create RNN
class RNN(nn.Module): #Check RNN inputs for classification task
    def __init__(self, batch_sizes = 32,numLayers = 2, numInputs = 3750,numNeurons = 29,numOutputs=11):
        super(RNN,self).__init__()

        self.numNeurons = numNeurons
        self.batch_sizes = batch_sizes
        self.conv = nn.Conv1d(in_channels = numInputs, out_channels = 500, kernel_size=5)
        self.relu = nn.LogSigmoid()
        self.pool = nn.MaxPool1d(kernel_size=5,stride=2)
        self.rnn = nn.RNN(input_size = 12500, hidden_size = numNeurons,num_layers = numLayers,nonlinearity='relu', batch_first = True)
        
        self.fc = nn.Linear(numNeurons,numOutputs)
        self.activation = nn.Softmax(dim = 0)
        
    def forward(self, x):
        # print(x.shape) #10*3750*29
        x = self.conv(x)
        x = self.relu(x)
        x = torch.reshape(x,(self.batch_sizes,-1))
        self.hidden = self.rnn(x)[0]
        out = self.fc(self.hidden)
        out = self.activation(out)
        
        return out

RNN_model = RNN()

error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.0001
optimizer = torch.optim.Adam(RNN_model.parameters(), lr=learning_rate)
#%%
#train model
frames = 29
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        batch_size = images.shape[0]
        images = torch.reshape(images,(batch_size,-1,29)) # 10*3750*29
        if batch_size == 32:
            # Clear gradients
            optimizer.zero_grad()
            
            outputs_RNN = RNN_model(images)
            # print(outputs_RNN.shape)
            loss = error(outputs_RNN[:,0], labels.float())
            predicted = torch.max(outputs_RNN, 1)[1]
            # Total number of labels
            correct += (predicted == labels).sum()
            train_acc = correct/len(labels)
            train_loss = loss.data
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()
        
            count += 1
    # Calculate Accuracy         
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in test_loader:
        # print(labels.shape)
        batch_size = images.shape[0]
        images = torch.reshape(images,(batch_size,-1,29))
        if batch_size == 32:

        
            outputs_RNN = RNN_model(images)
            # print(outputs_RNN.shape)
            # Get predictions from the maximum value
            predicted = torch.max(outputs_RNN, 1)[1]
            # Total number of labels
            total += len(labels)
            correct += (predicted == labels).sum()
        
    accuracy = 100 * correct / float(total)
    # store loss and iteration
    loss_list.append(loss.data)
    iteration_list.append(count)
    accuracy_list.append(accuracy)
    # Print Loss
    print('Epoch:{}--Train Loss:{}--Train Accuracy:{}--Valid Loss:{}--Valid Accuracy:{}%'.format(epoch, train_loss, train_acc, loss.data, accuracy))


# %%
