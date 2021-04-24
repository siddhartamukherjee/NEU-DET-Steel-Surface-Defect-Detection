#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils import data as torch_data
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import albumentations.pytorch as A
from torchvision import transforms


# In[2]:


class DataRetriever(torch_data.Dataset):
    def __init__(self, i_path, a_path):
        self.i_path = i_path
        self.a_path = a_path
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.transforms = self.get_transforms()
          
    def __len__(self):
        return len(self.i_path)
    
    def __getitem__(self, index):
        img = Image.open(self.i_path[index]).convert("RGB")
        #img = cv2.resize(img, (224, 224))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tree = ET.parse(self.a_path[index])
        root = tree.getroot()
        boxes = []
        masks = []
        for box in root.findall('.//bndbox'): 
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
            
        #extract image  dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
                
        # create one array for all masks, each on a different channel
        masks = np.zeros([height, width, len(boxes)], dtype='uint8')
                
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
        
        mask = transforms.ToPILImage()(masks) 
        mask = mask.convert('RGB')
        
        if self.transforms:
            image = self.transforms(img)
            image_mask = self.transforms(mask)
    
            
        return image, image_mask
    
    def get_transforms(self):
        return transforms.Compose([
            transforms.RandomRotation(10),      # rotate +/- 10 degrees
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            transforms.Resize(572),             # resize shortest side to 224 pixels
            #transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])


# In[ ]:




