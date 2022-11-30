import os
import natsort
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import CNN
import torch.nn as nn

frame_lists = np.ndarray([])
video_dir = r'/Users/liuxihan/Desktop/Lip_Reading/example_lip_extracted/VideoImageExample/frameExtraction/'
frames = natsort.natsorted(os.listdir(video_dir))
CNN_results = torch.Tensor()
convert_tensor = transforms.ToTensor()
CNN_model = CNN()

for frame in frames:
    frame_image = Image.open(os.path.join(video_dir,frame))
    
    image2tensor = convert_tensor(frame_image)
    # print(image2tensor.shape)
    # test = CNN()
    # cnn_result = test(image2tensor)
    np.append(frame_lists,image2tensor)

# print(convert_tensor)
video_tensor = convert_tensor(frame_lists)
print(video_tensor.shape)