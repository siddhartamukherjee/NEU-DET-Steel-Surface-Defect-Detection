## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Files and Folders](#files-and-folders)
* [Setup](#Setup)



# General info
The objective of the project is to build a pytorch segmentation model which can identify defect location on steel surface. The segmentation model used here is Unet with Resnet encoder. For this we have used the open surface defect database from Northeaster University(NEU).

In this database, six kinds of typical surface defects of the hot-rolled steel strip are collected, i.e., rolled-in scale (RS), patches (Pa), crazing (Cr), pitted surface (PS), inclusion (In) and scratches (Sc). The images provided in the database are in grayscale and of uniform dimensions 200x200.


![alt text](http://faculty.neu.edu.cn/yunhyan/Webpage%20for%20article/NEU%20surface%20defect%20database/Fig.2.png)


The database can be found in the below link:-

http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

The description of the database in the above url states that the database includes 1,800 grayscale images: 300 samples each of six different kinds of typical surface defects

The defect images are found in the folder named IMAGES. The folder named ANNOTATIONS contains the details of the defect location on each image in xml files.

Before we started with the project we ran the script named Create_Validation_Images.ipynb. This script created two folders named Validation_Images and Validation_Annotations. The script randomly selected five images from each class and moved those images and their annotation xmls to the folders Validation_Images and Validation_Annotations respectively. The model training won't be done on the images present in the Validation_Images. These images will be only used for final validation with our model.

The details of the key files and folders are given below.

## Technologies

The technologies used in this project are given below:-

* Python(v3.7.10)
* Pytorch(v1.8.1)


## Files and Folders

Below is the description of the important files and folders in the project:-

* **ANNOTATIONS** - This folder contains the annotation xmls of the samples on which model training and validation were done.
* **IMAGES** - This folder contains the sample images which were used for mode training and validation.
* **Validation_Annotations** - This folder contains the annotation xmls of the images which won't be used for model training and validation.
* **Validation_Images** - This folder contains the images which won't be used for model training and validation. The images from this folder are used for inferencing.
* **Models** - This folder contains the trained model's .pth file.
* **Utilities** - This folder contains some custom utility scripts that are required for model training, evalutation and inferencing. The python scripts present in this folder are listed below:-
  * **Meter.py** - This python script contains the calculation logic for the metrics used for model evaluation i.e IOU, Dice Coefficient, Dice Positive and Dice Negative.
  * **Trainer.py** - This python script contains logic to train the model. 
  * **Data_Retriever_Seg.py** - The output of this python script are Train and Test Datasets.
  * **Extract_masks.py** - This file has functions to extract masks from the annotation xmls.
  * **Resnet_Unet.py** - This python script contains the model class. The model returned by this script is a Unet model with Resnet encoder.
  * **Data_Retriever_Inference_Real_Time.py** - This script returns the data loader for inferencing.
* **Create_Validation_Images.ipynb** - This ipynb file creates two folders named Validation_Images and Validation_Annotations. The script randomly selected five images from each     class and moved those images and their annotation xmls to the folders Validation_Images and Validation_Annotations respectively. The model training won't be done on the images present in the Validation_Images. These images will be only used for final validation with our model.
* **Exploratory_Data_Analysis.ipynb** - This ipynb file gives an analysis of the data.
* **Train_Segmentation_Model_FPN+inceptionv4.ipynb** - This ipynb file is used to train an FPN with inceptionv4 encoder.
* **Train_Segmentation_Model_FPN+Resnet.ipynb** - This ipynb file is used to train an FPN with resnet34 encoder.
* **Train_Segmentation_Model_FPN+xception.ipynb** - This ipynb file is used to train an FPN with xception encoder.
* **Train_Segmentation_Model_Unet+Resnet.ipynb** - This ipynb file is used to train an Unet with resnet34 encoder.
* **Inference_Script.ipynb** - This ipynb file is used for inferencing. It reads the images from the Validation_Images folder as input and generates the masks as output. It also shows the comparison of the predicted mask with the original mask.

## Setup

This section gives an overview of how to use the project.

* Install the requirements from the requirements.txt file using the command **pip install -r requirements.txt**.
* First to get an overview about the data have a look at the ipynb file named Exploratory_Data_Analysis.ipynb.
* The respective model trainings can be done by the script Train_Segmentation_Model** scripts.
* Once, the model has been trained the script named Inference_Script.ipynb can be used to get inference on the validation images.
