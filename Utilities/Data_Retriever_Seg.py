#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils import data as torch_data
from torchvision import datasets, transforms, models
from PIL import Image
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from albumentations import (HorizontalFlip, Normalize, Compose, Resize)
from albumentations.pytorch import ToTensor
from torchvision import transforms
from itertools import product


# In[2]:


class DataRetriever(torch_data.Dataset):
    def __init__(self, df, image_folder, annot_folder, mean, std, phase):
        self.df = df
        self.mean = mean
        self.std = std
        self.phase = phase
        self.image_folder = image_folder
        self.annot_folder = annot_folder
        self.transforms = self.get_transforms()
        self.fnames = self.df.Name.tolist()
          
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        image_id, mask = self.make_mask(index)
        img = cv2.imread(image_id)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 1x200x200x6
        mask = mask[0].permute(2, 0, 1) # 6x200x200
        return img, mask
    
    
    def make_mask(self,row_id):
        name = self.df.iloc[row_id].Name
        annot = "\\".join([self.annot_folder, name+'.xml'])
        fname = "\\".join([self.image_folder, (name+'.jpg')])
        labels = self.df.iloc[row_id][1:7].to_dict()
        
        tree = ET.parse(annot)
        root = tree.getroot()
        #extract image  dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        masks = np.zeros((height, width, len(labels)), dtype=np.float32) # float32 is V.Imp
        
        for idx, label in enumerate(labels):
            if labels.get(label) == 1:
                boxes = []
                for obj in root.findall('.//object'):
                    if obj.find('name').text == label:                        
                        for box in obj.findall('.//bndbox'): 
                            xmin = int(box.find('xmin').text)
                            ymin = int(box.find('ymin').text)
                            xmax = int(box.find('xmax').text)
                            ymax = int(box.find('ymax').text)
                            coors = [xmin, ymin, xmax, ymax]
                            boxes.append(coors)    
                            
                mask = np.zeros((height, width), dtype=np.uint8)
                for i in range(len(boxes)):
                    box = boxes[i]
                    row_s, row_e = box[1], box[3]
                    col_s, col_e = box[0], box[2]
                    row_corr = [*range(row_s, row_e)]
                    col_corr = [*range(col_s, col_e)]
                    coords = np.array(list(product(row_corr, col_corr)))
                    mask[coords[:,0], coords[:,1]] = 1
                    
                masks[:, :, idx] = mask.reshape(height, width, order='F')
         
        return fname, masks
    
    def get_transforms(self):
        list_transforms = []
        if self.phase == "train":
            list_transforms.extend(
                [
                    HorizontalFlip(p=0.5), # only horizontal flip as of now
                ]
            )
        list_transforms.extend(
            [
                Normalize(mean=self.mean, std=self.std, p=1),
                Resize(128,128),
                ToTensor(),
            ]
        )
        list_trfms = Compose(list_transforms)
        return list_trfms


# In[ ]:




