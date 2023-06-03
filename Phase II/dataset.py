# import required modules
import os
import glob 
import random

import nibabel as nib 
import numpy as np
from sklearn.preprocessing import MinMaxScaler 

import torch
from torch.utils.data import Dataset

# -------------------------------------------------------------------------------------- #
# Class:       MRIDataset                                                                #
# Description: This class is used to extract, preprocess, and return .nii files from     #
#              the data archive.                                                         #
# -------------------------------------------------------------------------------------- #
class MRIDataset(Dataset):
    # function:    init
    # description: initializes class attributes
    # inputs:      t1Dir - directory containing t1 .nii files
    #              t2Dir - directory containing t2 .nii files
    # outputs:     none
    def __init__(self, t1Dir, t2Dir, train=True):
        # if loading training data, shuffle the lists of images so that the data isn't paired
        if train:
            self.t1Images = glob.glob(f'{t1Dir}/BraTS20_Training_*/BraTS20_Training_*_t1ce.nii')
            self.t2Images = glob.glob(f'{t2Dir}/BraTS20_Training_*/BraTS20_Training_*_t2.nii')
            
            random.shuffle(self.t1Images)
            random.shuffle(self.t2Images)
            
        # if validation, sort the lists of images so we can compare reals to fakes
        else:
            self.t1Images = sorted(glob.glob(f'{t1Dir}/BraTS20_Validation_*/BraTS20_Validation_*_t1ce.nii'))
            self.t2Images = sorted(glob.glob(f'{t2Dir}/BraTS20_Validation_*/BraTS20_Validation_*_t2.nii'))
            
        self.t1Dir = t1Dir
        self.t2Dir = t2Dir
        self.length = len(self.t1Images)
        
    # function:    len
    # description: returns the length of the dataset (number of images of each contrast)
    # inputs:      none
    # outputs:     the length of the dataset
    def __len__(self):
        return self.length
    
    # function:    getitem
    # description: loads, preprocesses, and returns a single t1 and t2 image as np arrays
    # inputs:      index - indicates which images from the dataset to return
    # outputs:     a single t1 and t2 image as np arrays
    def __getitem__(self, index):
        # get the paths to the images 
        t1 = self.t1Images[index % self.length]        
        t2 = self.t2Images[index % self.length]

        # load the images from .nii files
        t1Image = np.array(nib.load(t1).get_fdata())
        t2Image = np.array(nib.load(t2).get_fdata())
        
        # the raw pixel values range from roughly 0 to 2000; we'll want to rescale them to be 0 to 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        t1Image = scaler.fit_transform(t1Image.reshape(-1, t1Image.shape[-1])).reshape(t1Image.shape)
        t2Image = scaler.fit_transform(t2Image.reshape(-1, t2Image.shape[-1])).reshape(t2Image.shape)

        # crop the images from 240x240 to 144x176 to reduce useless features (black space)
        t1Image = t1Image[48:192,38:214,75]
        t2Image = t2Image[48:192,38:214,75]

        # return the images as np arrays
        return t1Image.astype(np.float32), t2Image.astype(np.float32)
        
        