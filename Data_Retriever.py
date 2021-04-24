#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
from torch.utils import data as torch_data
from torchvision import datasets, transforms, models
from PIL import Image


# In[3]:


class DataRetriever(torch_data.Dataset):
    def __init__(self, path, categories=None):
        self.path = path
        self.categories = categories
        self.transforms = self.get_transforms()
          
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        img = Image.open(self.path[index])
        #img = cv2.resize(img, (224, 224))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(img)
        
        if self.categories is None:
            return img
        
        y = self.categories[index] 
        return img, y
    
    def get_transforms(self):
        return transforms.Compose([
            transforms.RandomRotation(10),      # rotate +/- 10 degrees
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            #transforms.Resize(224),             # resize shortest side to 224 pixels
            #transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])


# In[ ]:




