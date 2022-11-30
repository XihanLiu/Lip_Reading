"""
Dataloader of cropped mouth frame by frame.
"""
import os
import natsort
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import CNN
import torch.nn as nn
from model import RNN


gif_dir = r'/Users/liuxihan/Desktop/Lip_Reading/example_lip_extracted/GIFImageExample/Lip_frameByFrame/ABOUT_00001_31.gif/standard/typeI'
frames = natsort.natsorted(os.listdir(gif_dir))
CNN_results = torch.Tensor()
convert_tensor = transforms.ToTensor()
CNN_model = CNN()
RNN_model = RNN()

for frame in frames:
    frame_image = Image.open(os.path.join(gif_dir,frame))
    image2tensor = convert_tensor(frame_image)
    cnn_result = CNN_model(image2tensor)
    rnn_result = RNN_model(cnn_result.reshape(1,-1))
    print(rnn_result)
    
