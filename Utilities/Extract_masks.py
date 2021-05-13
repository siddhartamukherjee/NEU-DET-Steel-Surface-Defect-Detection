#!/usr/bin/env python
# coding: utf-8

# In[2]:


import xml.etree.ElementTree as ET
import os
import cv2
import pandas as pd
import numpy as np
from itertools import product


# In[ ]:

def create_filepaths(path):
    df = pd.DataFrame()
    for (dirpath, dirnames,filenames) in os.walk(path):
        for filename in filenames:
            temp_path = "\\".join([path, filename])
            tree = ET.parse(temp_path)
            root = tree.getroot()
            dict1 = dict()
            ls = []
            for description in root.iter('name'):
                ls.append(description.text)
            res = np.array(ls)
            res = np.unique(res)
            ls = res.tolist()
            dict1['Name'] = filename[:-4]
            for ele in ls:
                dict1[ele] = 1
            df = df.append(dict1, ignore_index= True)
    df = df.replace(np.nan, 0)
    df = df[['Name',  'crazing', 'patches', 'inclusion', 'pitted_surface',  'rolled-in_scale', 'scratches']]
    df['Number_of_Defects'] = df.drop('Name',axis=1).sum(axis=1) 
        
    return df  

def make_mask(annot, labels):
    tree = ET.parse(annot)
    root = tree.getroot()
    #extract image  dimensions
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    masks = np.zeros((height, width, len(labels)), dtype=np.uint8) # float32 is V.Imp
        
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
         
    return masks

