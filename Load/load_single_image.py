import torch
from torchvision import transforms
from PIL import Image
from model import CNN
from model import RNN
import torch.nn as nn

#Convert one input image feeding into the CNN - > RNN
#Convert input image to tensor
test_image = Image.open(r'/Users/liuxihan/Desktop/Lip_Reading/example_lip_extracted/SingleImageExample/Type1_standard.jpg')
convert_tensor = transforms.ToTensor()
tensor = convert_tensor(test_image)
print(tensor.shape)

test = CNN()
CNN_result = test(tensor)
print(CNN_result.shape)
CNN_result= CNN_result.reshape(1,100)


test_RNN = RNN()
RNN_result = test_RNN(CNN_result)
print(RNN_result.shape)




