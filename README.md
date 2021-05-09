## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)



# General info
The objective of the project is to build a pytorch segmentation model which can identify defect location on steel surface. The segmentation model used here is Unet with Resnet encoder. For this we have used the open surface defect database from Northeaster University(NEU).

In this database, six kinds of typical surface defects of the hot-rolled steel strip are collected, i.e., rolled-in scale (RS), patches (Pa), crazing (Cr), pitted surface (PS), inclusion (In) and scratches (Sc). The images provided in the database are in grayscale and of uniform dimensions 200x200.

The database can be found in the below link:-

http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

The description of the database in the above url states that the database includes 1,800 grayscale images: 300 samples each of six different kinds of typical surface defects

The defect images are found in the folder named IMAGES. The folder named ANNOTATIONS contains the details of the defect location on each image in xml files.

Before we started with the project we ran the script named Create _Validation_Images.ipynb. This script created two folders named Validation_Images and Validation_Annotations. The script randomly selected five images from each class and moved those images and their annotation xmls to the folders Validation_Images and Validation_Annotations respectively. The model training won't be done on the images present in the Validation_Images. These images will be only used for final validation with our model.

The details of the key files and folders are given below.

## Technologies

The technologies used in this project are given below:-

* Python(v3.7.3)
* Pytorch(v1.1.0)
