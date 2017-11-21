from __future__ import print_function, division
import torch
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
import random
from PIL import Image
from utils_tools import *
class NUS_WIDE_Dataset(Dataset):
    """NUS-WIDE dataset."""

    def __init__(self, trainval_dir, root_dir, train=True, transform=None):
        """
        Args:
            trainval_dir (string): Path to the train.txt and val.txt.
            root_dir (string): Directory with all the images.
            train (bool):  determine to use train.txt or val.txt
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.trainval_dir = trainval_dir
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        # now load image_list
        if self.train == True:
            self.image_list = read_list(osp.join(self.trainval_dir, 'train.txt'))
        else:
            self.image_list = read_list(osp.join(self.trainval_dir, 'val.txt'))

        random.shuffle(self.image_list)
        print("there are total {} images, the training phase is {}".format(len(self.image_list), self.train))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = osp.join(self.root_dir, str(self.image_list[idx].split()[0]) +'.jpg')
        image = Image.open(img_name)
        
        image = image.convert('RGB')  
        #print(image)
        targets = [float(x) for x in (self.image_list[idx].split()[1:])]
        targets = np.array(targets)        
       
        #targets = targets.reshape(-1,1)
        #print(targets)
        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 'targets': targets}
        
        return sample
        
class NUS_WIDE_Dataset_Test(Dataset):
    """NUS-WIDE dataset."""

    def __init__(self, test_dir, root_dir, transform=None):
        """
        Args:
            trainval_dir (string): Path to the train.txt and val.txt.
            root_dir (string): Directory with all the images.
            train (bool):  determine to use train.txt or val.txt
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.test_dir = test_dir
        self.root_dir = root_dir
        self.transform = transform
        
        # now load image_list
        
        self.image_list = read_list(osp.join(self.test_dir, 'test.txt'))
        
        print("there are total {} images".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = osp.join(self.root_dir, str(self.image_list[idx].split()[0]) +'.jpg')
        image = Image.open(img_name)
        image = image.convert('RGB')  
        #print(image)
        targets = [float(x) for x in (self.image_list[idx].split()[1:])]
        targets = np.array(targets)

        #targets = targets.reshape(-1,1)
        #print(targets)
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'targets': targets}

        return sample


