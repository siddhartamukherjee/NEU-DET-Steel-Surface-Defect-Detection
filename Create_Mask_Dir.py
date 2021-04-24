#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import argparse
import shutil
import numpy as np
from lxml import etree
from tqdm import tqdm


# In[2]:


def dir_create(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)


# In[3]:


def parse_args():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Convert CVAT XML annotations to contours'
        )
    parser.add_argument(
        '--image-dir', metavar='DIRECTORY', required=True,
        help='directory with input images'
        )
    parser.add_argument(
        '--cvat-xml', metavar='FILE', required=True,
        help='input file with CVAT annotation in xml format'
        )
    parser.add_argument(
        '--output-dir', metavar='DIRECTORY', required=True,
        help='directory for output masks'
        )
    parser.add_argument(
        '--scale-factor', type=float, default=1.0,
        help='choose scale factor for images'
        )
    return parser.parse_args()


# In[4]:


def parse_anno_file(cvat_xml, image_name):
    root = etree.parse(cvat_xml).getroot()
    anno = []
    image_name_attr = ".//image[@name='{}']".format(image_name)
    for image_tag in root.iterfind(image_name_attr):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)
        for box_tag in image_tag.iter('box'):
            box = {'type': 'box'}
            for key, value in box_tag.items():
                box[key] = value
            box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
                box['xtl'], box['ytl'], box['xbr'], box['ybr'])
            image['shapes'].append(box)
        image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)
    return anno


# In[5]:


def create_mask_file(width, height, bitness, background, shapes, scale_factor):
    mask = np.full((height, width, bitness // 8), background, dtype=np.uint8)
    for shape in shapes:
        points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points*scale_factor
        points = points.astype(int)
        mask = cv2.drawContours(mask, [points], -1, color=(255, 255, 255), thickness=5)
        mask = cv2.fillPoly(mask, [points], color=(0, 0, 255))
    return mask


# In[6]:


def main():
    args = parse_args()
    dir_create(args.output_dir)
    img_list = [f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
    mask_bitness = 24
    for img in tqdm(img_list, desc='Writing contours:'):
        img_path = os.path.join(args.image_dir, img)
        anno = parse_anno_file(args.cvat_xml, img)
        background = []
        is_first_image = True
        for image in anno:
            if is_first_image:
                current_image = cv2.imread(img_path)
                height, width, _ = current_image.shape
                background = np.zeros((height, width, 3), np.uint8)
                is_first_image = False
            output_path = os.path.join(args.output_dir, img.split('.')[0] + '.png')
            background = create_mask_file(width,
                                          height,
                                          mask_bitness,
                                          background,
                                          image['shapes'],
                                          args.scale_factor)
        cv2.imwrite(output_path, background)


# In[7]:


#if __name__ == "__main__":
    #main()


# In[ ]:




