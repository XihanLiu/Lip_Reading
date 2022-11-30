#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import torch as torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

def fit(model, trainDataLoader, valDataLoader, n_epochs, batch_size, lr):
    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=lr)
    lossFn = nn.NLLLoss()
    H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # loop over our epochs
    for e in range(0, n_epochs):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0

        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0

        # loop over the training set
        for (x, y) in trainDataLoader:
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFn(pred, y)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in valDataLoader:
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValLoss += lossFn(pred, y)
                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)
        # update our training history
        H["train_loss"].append(avgTrainLoss)
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, n_epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

