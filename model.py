import os
import torch as torch
import torch.nn as nn


#suppose input image size = 3 * 50 * 100
class CNN(nn.Module):
    def __init__(self, numChannels=3, numClasses=100):
        """
        input: 
           numChannels: 3(RGB image)
           Height: 50
           Width:100
           numClasses: output features of CNN

        """
        # Parent constructor
        super(CNN,self).__init__()

        #input:3 * 50 * 100
        #First layer: Conv2d => Relu => MaxPool2d
        self.conv1=nn.Conv2d(in_channels=numChannels,out_channels=20,kernel_size=5)
        #output:((50-5)+1)*((100-5)+1) => 20 * 46 * 96
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        #output: 20 * 23 * 48
        
        #Second layer: Conv2d => Relu => MaxPool2d
        self.conv2=nn.Conv2d(in_channels=20,out_channels=50, kernel_size=5)
        #output: 50 * 19 * 44
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        #output: 50 * 9 * 22

        #First FC layer => ReLu layer
        self.fc1 = nn.Linear(in_features=9900,out_features=1000)
        self.relu3 = nn.ReLU()

        #Second FC layer => ReLu layer
        self.fc2 = nn.Linear(in_features=1000,out_features=500)
        self.relu4 = nn.ReLU()

        #Softmax Layer
        self.fc3 = nn.Linear(in_features=500, out_features=numClasses)
        self.Softmax = nn.Softmax(dim=0)

    def forward(self,x):

        #CNN branch
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = torch.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = torch.flatten(x) 
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        out = self.Softmax(x) #100*1

        return out

class RNN(nn.Module): #Check RNN inputs for classification task
    def __init__(self,batch_sizes = 5,numLayers = 1, numInputs = 100,numNeurons = 20,numOutputs=10):
        super(RNN,self).__init__()

        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.numNeuros = numNeurons
        self.batch_sizes = batch_sizes
        self.rnn = nn.RNN(numInputs, numNeurons,numLayers)
        self.fc = nn.Linear(numNeurons,numOutputs)
    
    def init_hidden(self):
        self.hidden_states = None

    def forward(self, x):
        """
        Calculate the output.
        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).
        Returns
        ----------
        output: ``torch.FloatTensor``.   
            The output of RNNs.
        """
        # transforms X to dimensions: n_steps X batch_size X n_inputs

        self.hidden = self.init_hidden()
        lstm_out,self.hidden = self.rnn(x,self.hidden)
        out = self.fc(self.hidden)

        return out


