import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import csv

single_video_frames_dir = './example_lip_extracted/VideoImageExample/Lip_frameByFrame/'
# frames_list = natsort.natsorted(os.listdir(single_video_frames_dir))

class LipDataset(Dataset):
    def __init__(self,img_dir):
        self.img_dir = img_dir
        self.surfix = '.mp4/standard/typeI/'

    def __getlabels__(self):
        """
        Create a single label file.
        """
        Label_path = './example_lip_extracted/VideoImageExample/Labels.csv'
        with open(Label_path,'w') as f:
            labels_dir = os.listdir(self.img_dir)
            length = len(labels_dir)
            for label in range(1,length):
                name = os.path.splitext(labels_dir[label])[0]
                writer =  csv.writer(f)
                writer.writerow([name])
            
        f.close()
        return Label_path 
    
    def getitems(self, index):
        """
        Input: Index of label in the labels.
        Return: Frames of image with label.
        """
        with open(self.__getlabels__(),'r') as f:
            contents = f.readlines()
            label = (contents[index])[0:-1]
            images = os.listdir(os.path.join(self.img_dir,label+self.surfix))
            
        f.close()
        return (images, label)
Output = LipDataset(single_video_frames_dir)
Frames = Output.getitems(0)
print(Frames)