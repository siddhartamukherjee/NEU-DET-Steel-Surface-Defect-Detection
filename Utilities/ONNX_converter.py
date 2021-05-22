#!/usr/bin/env python
# coding: utf-8

# In[2]:


from Resnet_UNet import ResNetUNet
import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision import models
import torch
from collections import OrderedDict
import argparse


# In[2]:


def main():
    my_parser = argparse.ArgumentParser(description='Convert pytorch model to ONNX format.')
    
    my_parser.add_argument('Model',
                       metavar='model',
                       type=str,
                       help='The model to be converted.')
    
    my_parser.add_argument('Encoder',
                       metavar='encoder',
                       type=str,
                       help='The encoder for the model.')
    
    my_parser.add_argument('Encoder_Weights',
                       metavar='encoder_weights',
                       type=str,
                       help='The encoder weights for the model.')
    
    my_parser.add_argument('Classes',
                       metavar='classes',
                       type=str,
                       help='The number of classes the model can predict.')
        
    my_parser.add_argument('Model_Path',
                       metavar='model_path',
                       type=str,
                       help='The path to the saved pytorch model.')
      
    
    my_parser.add_argument('Output_Path',
                       metavar='output_path',
                       type=str,
                       help='The path where the onnx model will be saved.')
    
    args = my_parser.parse_args()
    
    model_name = args.Model
    ENCODER = args.Encoder
    ENCODER_WEIGHTS = args.Encoder_Weights
    CLASSES = int(args.Classes)
    ckpt_path = args.Model_Path
    output_path = args.Output_Path
    print(f"Model name :{model_name}")
    print(f"Model checkpoint path is :{ckpt_path}")
    print(f"Model output path is :{output_path}")
    
    ACTIVATION = None
  
    
    if model_name == 'Unet':
        # create segmentation model with pretrained encoder
        model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            in_channels=3,
            classes=CLASSES, 
            activation=ACTIVATION,
            )
    elif model_name == 'FPN':
        # create segmentation model with pretrained encoder
        model = smp.FPN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            in_channels=3,
            classes=CLASSES, 
            activation=ACTIVATION,
            )
        
    elif model_name == 'DeepLab_v3':
        # create segmentation model with pretrained encoder
        model = smp.DeepLabV3(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            in_channels=3,
            classes=CLASSES, 
            activation=ACTIVATION,
            )
    else:
        print("Unknown model. Please try other model.")
        exit()
        
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
                  output_path, # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names
    
    print("Successfully converted the model to onnx format.")


# In[ ]:


if __name__ == "__main__":
    main()

