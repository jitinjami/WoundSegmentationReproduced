# -*- coding: utf-8 -*-

'''
The 'data' folder is divided into 4 sub-folder:
1. external - Here the images and masks are raw and require some processing 
        before they can be moved to interim. Datasets are divided.
2. raw - Here the data is in the exact format we need them, image and mask.
        Datasets are divided.
3. interim - Here images and masks are places after augmentation. Datasets are divided.
4. processed - Here images and masks are places in their respective folder. 
            Datasets are combined, if both are used.
'''

#import os
#import glob
import shutil
import random
#from pathlib import Path
#import cv2
#import albumentations as A
from src.data.dataset_processing import WoundDataset, WSegDataset
from src.utils import create_empty_folder

def create_interim_dir(data_path: str, raw_dataset: WoundDataset):
    
    # Splitting the images into [85:10:5] ratio for [train:val:test]
    val_length = round(raw_dataset.len*0.1)
    test_length = round(raw_dataset.len*0.05)

    #Split the names
    val_names = random.sample(raw_dataset.image_names, val_length)
    test_names = random.sample(set(raw_dataset.image_names)-set(val_names), test_length)
    #train_names = list(set(raw_dataset.image_names) - set(val_names)-set(test_names))

    #Create empty folders if it doesn't exist
    create_empty_folder(data_path + 'interim/train/images/')
    create_empty_folder(data_path + 'interim/train/masks/')

    create_empty_folder(data_path + 'interim/val/images/')
    create_empty_folder(data_path + 'interim/val/masks/')

    create_empty_folder(data_path + 'interim/test/images/')
    create_empty_folder(data_path + 'interim/test/masks/')

    #First move everything to train folder
    raw_dataset.move(dest_path = data_path + 'interim/train/')
    train_dataset = WoundDataset(home_path=data_path + 'interim/train/')

    #Move images and masks to validation folder
    for image_name in val_names:
        index = train_dataset.image_names.index(image_name)
        shutil.move(str(train_dataset.image_paths[index]), data_path + 'interim/val/images/')
        shutil.move(str(train_dataset.mask_paths[index]), data_path + 'interim/val/masks/')
    
    #Move images and masks to test folder
    for image_name in test_names:
        index = train_dataset.image_names.index(image_name)
        shutil.move(str(train_dataset.image_paths[index]), data_path + 'interim/test/images/')
        shutil.move(str(train_dataset.mask_paths[index]), data_path + 'interim/test/masks/')
    
    train_dataset = WoundDataset(home_path=data_path + 'interim/train/')
    val_dataset = WoundDataset(home_path=data_path + 'interim/val/')
    test_dataset = WoundDataset(home_path=data_path + 'interim/test/')
    return train_dataset, val_dataset, test_dataset


def make_dataset1(data_path:str, ws_aug: bool):
    '''
    Process images and masks form dataset1 from external until interim
    Inside the external data folder:
    Dataset1 - Wound Segmentation Dataset
    Input:
        data_path - path to where the datasets are stored
    '''

    # Move cropped + padded images and masks to raw
    print("Processing MobilenetV2 dataset and moving cropped+padded images to 'raw' folder")
    external_dataset = WoundDataset(data_path + 'external/dataset1/')
    external_dataset.move(data_path + 'raw/dataset1/')

    # Split images in raw folder in train, test, val and move to interim folder
    print("Create Validation and Test datasets")
    raw_dataset = WoundDataset(data_path + 'raw/dataset1/')
    train_dataset, val_dataset, test_dataset = create_interim_dir(data_path, raw_dataset)

    # Create augmentations on train dataset
    print("Creating augmentation on train dataset and moving to 'processed' folder")
    train_dataset.augment(save_path = data_path + 'processed/train/', ws_aug=ws_aug)

    # Move val and test datasets to processed folder
    print("Moving validation and test dataset to 'processed' folder")
    val_dataset.move(data_path + 'processed/val/')
    test_dataset.move(data_path + 'processed/test/')


def make_dataset2(data_path:str, ws_aug: bool):
    '''
    Process images and masks form dataset2 from external until interim
    Inside the external data folder:
    Dataset2 - WSeg Dataset (images are already cropped)
    Input:
        data_path - path to where the datasets are stored
    '''

    # Create padded images and masks and move to 'raw'
    print("Processing WSeg dataset and moving cropped+padded images to 'raw' folder")
    external_dataset = WSegDataset(data_path + 'external/dataset2/')
    external_dataset.process_wseg()
    external_dataset.move(dest_path=data_path+'raw/dataset2/', 
                          images_path=external_dataset.padded_images_path, 
                          masks_path=external_dataset.padded_masks_path)
    
    # Split images in raw folder in train, test, val and move to interim folder
    print("Create Validation and Test datasets")
    raw_dataset = WoundDataset(data_path + 'raw/dataset2/')
    train_dataset, val_dataset, test_dataset = create_interim_dir(data_path, raw_dataset)

    # Create augmentations on train dataset
    print("Creating augmentation on train dataset and moving to 'processed' folder")
    train_dataset.augment(save_path = data_path + 'processed/train/', ws_aug=ws_aug)

    # Move val and test datasets to processed folder
    print("Moving validation and test dataset to 'processed' folder")
    val_dataset.move(data_path + 'processed/val/')
    test_dataset.move(data_path + 'processed/test/')    

if __name__=='__main__':
    pass
