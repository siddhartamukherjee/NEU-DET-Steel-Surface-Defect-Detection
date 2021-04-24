#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[2]:


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = self.double_conv(1,64)
        self.down_conv_2 = self.double_conv(64,128)
        self.down_conv_3 = self.double_conv(128,256)
        self.down_conv_4 = self.double_conv(256,512)
        self.down_conv_5 = self.double_conv(512,1024)
        
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, 
                                             out_channels=512,
                                             kernel_size=2,
                                             stride=2)
        
        self.up_conv_1 = self.double_conv(1024,512)
        
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, 
                                             out_channels=256,
                                             kernel_size=2,
                                             stride=2)
        
        self.up_conv_2 = self.double_conv(512,256)
        
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, 
                                             out_channels=128,
                                             kernel_size=2,
                                             stride=2)
        
        self.up_conv_3 = self.double_conv(256,128)
        
        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, 
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=2)
        
        self.up_conv_4 = self.double_conv(128,64)
        
        self.out = nn.Conv2d(in_channels=64,
                            out_channels=2,
                            kernel_size=1)
    
    def forward(self, image):
        #bs, c, h, w
        #encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        
        #decoder
        x = self.up_trans_1(x9)
        y = self.crop_img(x7,x)
        x = self.up_conv_1(torch.cat([x,y], 1))

        x = self.up_trans_2(x)
        y = self.crop_img(x5,x)
        x = self.up_conv_2(torch.cat([x,y], 1))
        
        x = self.up_trans_3(x)
        y = self.crop_img(x3,x)
        x = self.up_conv_3(torch.cat([x,y], 1))
        
        x = self.up_trans_4(x)
        y = self.crop_img(x1,x)
        x = self.up_conv_4(torch.cat([x,y], 1))
        
        x = self.out(x)
        return x


    def double_conv(self, in_c,out_c):
        conv = nn.Sequential(nn.Conv2d(in_c,out_c,kernel_size=3),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_c,out_c,kernel_size=3),
                        nn.ReLU(inplace=True))
        return conv
    
    def crop_img(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


# In[ ]:





# In[ ]:




