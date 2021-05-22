#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR
import torch.backends.cudnn as cudnn
import time
import gc
from tqdm import tqdm
import os
from Utilities.Extract_masks import create_filepaths
from Utilities.Data_Retriever_Seg import DataRetriever
from Utilities.Meter import Meter


# In[2]:


class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, lr, epochs, path):
        self.num_workers = 6
        self.batch_size = {"train": 4, "val": 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = lr
        self.num_epochs = epochs
        self.epochs_passed = 0
        self.best_loss = float("inf")
        self.best_dice = float(0)
        self.phases = ["train", "val"]
        CUDA_VISIBLE_DEVICES=0,1 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", patience=3, verbose=True)
        #self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)
        #intialize the model
        self.net= nn.DataParallel(self.net)
        #self.net = self.net.share_memory()
        self.net = self.net.to(self.device)
        self.PATH = path
        if os.path.exists(self.PATH):
            checkpoint = torch.load(self.PATH)
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epochs_passed = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
        cudnn.benchmark = True
        self.dataloaders = {
            phase: self.provider(
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        #if phase == "train":
            #total_batches = 100
        #else:
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        torch.cuda.empty_cache()
        #num_batch = 0
        for itr, batch in enumerate(tk0): # replace `dataloader` with `tk0` for tqdm
            #num_batch = num_batch + 1
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                #if num_batch == total_batches:
                    #break
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
            tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = meter.epoch_log(epoch_loss, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss, dice, iou

    def start(self):
        for epoch in range(self.epochs_passed,self.num_epochs):
            print("Learning Rate = ",self.optimizer.param_groups[0]["lr"])
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss, val_dice, val_iou = self.iterate(epoch, "val")
                self.scheduler.step(val_dice)
            #if val_loss < self.best_loss:
            if val_dice > self.best_dice:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                state["best_dice"] = self.best_dice = val_dice
                torch.save(state, self.PATH)
            print()
            
    def provider(self,
        phase,
        mean=None,
        std=None,
        batch_size=8,
        num_workers=4,
    ):
        '''Returns dataloader for the model training'''
        image_folder = '.\IMAGES'
        annot_folder = '.\ANNOTATIONS'
        df = create_filepaths(annot_folder)
        train_df, val_df = train_test_split(df, test_size=0.3, stratify=df["Number_of_Defects"], random_state=69)
        df = train_df if phase == "train" else val_df
        image_dataset = DataRetriever(df, image_folder, annot_folder, mean, std, phase)
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,   
        )

        return dataloader
    
