"""
Dataloader of cropped mouth frame by frame.
Input: Video frames with labels in the folder name.
Output: Return the Database of training, validation, testing
"""
import os
import natsort
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn



gif_dir = './example_lip_extracted/VideoImageExample/Lip_frameByFrame/AFTERNOON.mp4/standard/typeI/'
frames = natsort.natsorted(os.listdir(gif_dir))
print(frames)
# CNN_results = torch.Tensor()
# convert_tensor = transforms.ToTensor()
# CNN_model = CNN()
# RNN_model = RNN()

# for frame in frames:
#     frame_image = Image.open(os.path.join(gif_dir,frame))
#     image2tensor = convert_tensor(frame_image)
#     print(rnn_result)
    
