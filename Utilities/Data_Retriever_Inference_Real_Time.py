#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pdb
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import (Normalize, Compose, Resize)
from albumentations.pytorch import ToTensor
import torch.utils.data as data


# In[2]:


class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, fname, mean, std):
        self.root = root
        self.fname = fname
        self.num_samples = 1
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                Resize(128,128),
                ToTensor()
            ]
        )

    def __getitem__(self, idx):
        fnames = self.fname
        path = os.path.join(self.root, fnames)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fnames, images

    def __len__(self):
        return self.num_samples


# In[ ]:





# In[ ]:




