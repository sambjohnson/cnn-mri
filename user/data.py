import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

# these are required for defining the regession ImageFolder
from typing import Dict, Any
from torchvision import datasets

import pandas as pd
import numpy as np
import os

class ToFloat(object):
    """ Converts the datatype in sample to torch.float32 datatype.
        - helper function to be used as transform (typically from uint8 to float)
        - useful because inputs must have the same datatype as weights of the n.n.
    """

    def __call__(self, target):
        target_tensor = torch.tensor(target)
        return target_tensor.to(torch.float32)


class ToRGB(object):
    """ Converts a 1-channel tensor into 3 (equal) channels
        for ease of use with pretrained vision models.
    """
    
    def __call__(self, image):
        image = torch.tensor(image)
        image = image.repeat(3, 1, 1)
        # print(image.shape) # for testing
        return image


class CustomImageDataset(Dataset):
    """ Custom dataset, like ImageFolder, works with arbitrary image labels.
        Labels should be a .csv in the format:
            image1filename.png, image1label
            image2filename.png, image2label
            ...
        Useful for regression; circumvents ImageFolder classification scheme
        which requires that images be sorted into subfolders corresponding to class names.
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
