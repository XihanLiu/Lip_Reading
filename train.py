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
from model import CNN,RNN
from fit import fit

# Create CNN
CNN_model = CNN()
RNN_model = RNN()
# model.cuda()
print(CNN_model)
print(RNN_model)
#%%
PATH_ROOT = '/Users/liuxihan/Desktop/Lip_Reading/Data/Lip_frameByFrame' # path to the 3d_matrix root
DatasetSize = "standard"
DatasetType = "I"
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
                        if ".jpg" in image_id and image_id[0] != ".":
                            current_image = plt.imread(image_folder_dirc+'/'+image_id)
                            if m == 0:
                                single_image_num_rows, single_image_num_cols, single_image_num_chennels =  current_image.shape
                                same_label_image = np.ones((single_image_num_chennels, single_image_num_rows, single_image_num_cols,num_frames)).astype('uint8')
                            for q in range(single_image_num_chennels):
                                same_label_image[q,:,:,m] = current_image[:,:,q]
                        else:
                            os.remove(Label_root+'/'+video_id)
                    if j < num_train:
                        TrainData_list.append(same_label_image)
                    else: 
                        TestData_list.append(same_label_image)
            else: 
                os.remove(Label_root+'/'+video_id)
        
    TrainData_array = np.array(TrainData_list).astype("uint8")
    TestData_array = np.array(TestData_list).astype("uint8")
    

    return TrainData_array, TestData_array, train_label[:,0], test_label[:,0], label_string_list
    
#%% test field
X_train, X_test, targets_train, targets_test, label_string_list = constuct_Dataset_withSplitingRatio(PATH_ROOT, Data_dirc, DatasetType, 0.9)
#%%
#convert all the variables to pytorch tensor format
#X_train should have shape (num_of_dataset,50,100,29,3) and targets_train(num_of_dataset,1)
#X_test should have shape (num_of_test_dataset,50,100,29,3) and targets_test(num_of_test_dataset,1)
train_x=torch.from_numpy(X_train).float()#change the array into Tensor
# del X_train
train_y=torch.from_numpy(targets_train).long()
# del targets_train
test_x=torch.from_numpy(X_test).float()
# del X_test
test_y=torch.from_numpy(targets_test).long()
# del targets_test

#%%
batch_size=1#the batch_size we will use for the training

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
n_iters = 2000
num_epochs = 20
#num_epochs = n_iters / (len(train_x) / batch_size)
#num_epochs = int(num_epochs)

# Create CNN
CNN_model = CNN()
RNN_model = RNN()
# model.cuda()
print(CNN_model)
print(RNN_model)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.0001
optimizer = torch.optim.Adam(CNN_model.parameters(), lr=learning_rate)
#%%
#train model
# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        # batch_size = images.shape[0]
        images = torch.squeeze(images)
        # if batch_size == 10:
        # train = Variable(images.view(10,3,50,100,29))
        
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        Outputs_Tensor = torch.FloatTensor(1,29) #(num_classes * frames)
        for j in range(images.shape[3]):
        # Forward propagation
            outputs_CNN = CNN_model(images[:,:,:,j])
            Outputs_Tensor[:,j] = outputs_CNN
        outputs_RNN = RNN_model(Outputs_Tensor)
        # Calculate softmax and ross entropy loss
        loss = error(outputs_RNN, labels)
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
                # batch_size = images.shape[0]
                images = torch.squeeze(images)
                # if batch_size == 10: 
                    # test = Variable(images.view(batch_size,3,50,100,29))
                    # test = Variable(images)
                    # print(test.shape)
                    # Forward propagation
                Outputs_Tensor = torch.FloatTensor(1,29) #(num_classes * frames)
                for j in range(images.shape[3]):
                    # Forward propagation
                    outputs_CNN = CNN_model(images[:,:,:,j])
                    Outputs_Tensor[:,j] = outputs_CNN
                    outputs_RNN = RNN_model(Outputs_Tensor)
                # Get predictions from the maximum value
                predicted = torch.max(outputs_RNN, 1)[1]
                
                # Total number of labels
                total += len(labels)
                # print(total)
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


# %%
