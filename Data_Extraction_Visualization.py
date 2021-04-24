#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xml.etree.ElementTree as ET
from os import walk
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class XMLExtraction:
    def __init__(self,path):
        self.path = path
        self.df = self.create_filepaths()
        
    def create_filepaths(self):
        f = []
        labels = []
        f_paths = []
        a_paths = []
        
        for (dirpath, dirnames,filenames) in walk(self.path):
            for filename in filenames:
                temp_path = "\\".join([self.path, filename])
                tree = ET.parse(temp_path)
                root = tree.getroot()
                for description in root.iter('name'):
                    label = description.text
                labels.append(label)
                for description in root.iter('filename'):
                    image_path = description.text
                    if image_path[-4:] != '.jpg':
                        image_path = image_path + '.jpg'
                    f_paths.append(("\\".join(['.\IMAGES', image_path])))
                file = filename[:-len('.xml')]
                f.append(file)
                a_paths.append("\\".join([self.path, filename]))
                
        
        df = pd.DataFrame(list(zip(f, f_paths, a_paths, labels)),
               columns =['Name', 'image_path', 'annotation_path', 'label'])
                
        return df   
    


# In[3]:


class Visualize:
    
    def __init__(self, f_path=None, a_path=None, name=None):
        self.f_path = f_path
        self.a_path = a_path
        self.name = name
    
    def extract_coordinates(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        X_min_corr = []
        for description in root.iter('xmin'):
            X_min_corr.append(int(description.text))
    
        X_max_corr = []
        for description in root.iter('xmax'):
            X_max_corr.append(int(description.text))
    
        Y_min_corr = []
        for description in root.iter('ymin'):
            Y_min_corr.append(int(description.text))
    
        Y_max_corr = []
        for description in root.iter('ymax'):
            Y_max_corr.append(int(description.text))
            
        return X_min_corr, X_max_corr, Y_min_corr, Y_max_corr
    
    def show_defect(self, f_path, a_path, name, color):
        X_min, X_max, Y_min, Y_max = self.extract_coordinates(a_path)
        image = cv2.imread(f_path)


        for xmin, xmax, ymin, ymax in zip(X_min, X_max, Y_min, Y_max):
            
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            #color = (0, 0, 255)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)

            black = [0,0,0]     #---Color of the border---
            constant=cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=black )

            white= np.zeros((100, constant.shape[1], 3), np.uint8)
            white[:] = (255, 255, 255) 
            vcat = cv2.vconcat((white, constant))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vcat,name,(30,50), font, 1,(0,0,0), 3, 0)

        plt.imshow(vcat)
        plt.show()
        
        


# In[ ]:




