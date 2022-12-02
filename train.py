import os
import torch
import torch.nn as nn
from torch.optim import Adam
from model import CNN,RNN
from dataset import LipDataset



train_dir = './Data/Lip_frameByFrame/'
dataset = LipDataset(train_dir)
numClasses = dataset.size()

CNN_Model = CNN()
RNN_Model = RNN()
for group in range(numClasses):
    names,label,length = dataset.getnames(group)
    print("Load in CNN model...")
    
    for frame in range(length):
        train_tensor,train_label = dataset.getitems(group,frame)
        CNN_Features = CNN_Model(train_tensor)
        
        #print(train_tensor.shape,train_label) # Check output: "torch.Size([3, 50, 100]) AROUND"