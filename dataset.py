import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import csv

single_video_frames_dir = './example_lip_extracted/VideoImageExample/Lip_frameByFrame/'

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
                writer = csv.writer(f)
                writer.writerow([name])
            
        f.close()
        return Label_path 
    
    def __len__(self):
        with open(self.__getlabels__(),'r') as f:
            lines = f.readlines()
            length = len(lines)
        f.close()
        return length
    
    def size(self):
        return self.__len__()

    def getnames(self, index):
        """
        Input:      Index of label in the labels.
        Return:     Frame names of video with label and num of frames.
        """
        with open(self.__getlabels__(),'r') as f:
            contents = f.readlines()
            label = (contents[index])[0:-1]
            images_names = os.listdir(os.path.join(self.img_dir,label+self.surfix))
            num_images = len(images_names)
            for index in range(len(images_names)):
                images_names[index] = os.path.join(self.img_dir,label+self.surfix,images_names[index])
        f.close()
        return images_names,label,num_images

    def getitems(self,index_group, index_single_frame):
        """
        Input:      Index of group of frames.
        Return:     tensor of one frame with label.
        """
        convert_tensor = transforms.ToTensor()
        frames, label, num_frames = self.getnames(index_group)
        select_frame = Image.open(frames[index_single_frame])
        tensor = convert_tensor(select_frame)
        return tensor,label

Output = LipDataset(single_video_frames_dir)
Size = Output.size()

for group in range(Size):
    names,label,length = Output.getnames(group)
    for frame in range(length):
        train_tensor,train_label = Output.getitems(group,frame)
        