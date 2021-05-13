#!/usr/bin/env python
# coding: utf-8

# In[2]:


from Resnet_UNet import ResNetUNet
import torch.nn as nn
from torchvision import models
import torch
from collections import OrderedDict
import argparse


# In[2]:


def main():
    my_parser = argparse.ArgumentParser(description='Convert pytorch model to ONNX format.')
    
    my_parser.add_argument('Model_Path',
                       metavar='model_path',
                       type=str,
                       help='The path to the saved pytorch model.')
    
    args = my_parser.parse_args()
    
    ckpt_path = args.Model_Path
    model = ResNetUNet(n_class=6, base_model=models.resnet18(pretrained=True))
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state["state_dict"].items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    input_var = torch.rand(1, 3, 128, 128)  # Use half of the original resolution.
    batch_size = 5
    # Export the model
    torch.onnx.export(model,                 # model being run
                  input_var,                 # model input (or a tuple for multiple inputs)
                  "./ONNX_models/Unet.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names
    
    print("Successfully converted the model to onnx format.")


# In[ ]:


if __name__ == "__main__":
    main()

