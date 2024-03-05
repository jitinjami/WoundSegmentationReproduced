import os
import glob
from pathlib import Path
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from yacs.config import CfgNode as CN
from src.utils import *

class ProcessedWoundDataset(Dataset):
    '''
    Dataset class the way pytorch likes it
    '''
    def __init__(self, home_path:str):
        self.imgs_path = home_path + 'images/'
        self.masks_path = home_path + 'masks/'

        self.image_paths = get_list_of_paths(self.imgs_path)
        self.mask_paths = get_list_of_paths(self.masks_path)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        #Make it 3D
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        # Make sure its (channel, height, width)
        return_image = torch.permute(torch.from_numpy(image).float(),(2,0,1))
        return_mask = torch.permute(torch.from_numpy(mask.astype(bool)).float(),(2,0,1))
        return return_image, return_mask
