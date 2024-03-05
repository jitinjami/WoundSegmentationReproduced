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

import os
#import glob
import shutil
import random
from pathlib import Path
import cv2
import albumentations as A
from src.utils import get_list_of_paths, get_list_of_file_names, create_empty_folder

class WoundDataset:
    '''
    Class definition for wound dataset. At each folder level, wound dataset will be its own instance
    '''
    def __init__(self, home_path: str):
        self.home_path = home_path

        self.images_path = self.home_path + 'images/'
        self.masks_path = self.home_path + 'masks/'
        
        self.image_paths = get_list_of_paths(self.images_path)
        self.image_names = get_list_of_file_names(self.images_path)

        self.mask_paths = get_list_of_paths(self.masks_path)
        self.mask_names = get_list_of_file_names(self.masks_path)

        if self.image_names != self.mask_names:
            print('Images and masks are not same')

        self.len = len(self.image_names)
    
    def move(self, dest_path: str, images_path = None, masks_path = None):
        '''
        Moves images and masks from self.home_path to dest_path if 'images_path' or
        'masks_path' are not provided in the arguments.
        It creates a new destination folder if it doesn't exist and also 
        creates an 'images' and 'masks' folder inside.
        '''

        if images_path is None:
            images_path = self.images_path
        
        if masks_path is None:
            masks_path = self.masks_path
        
        create_empty_folder(dest_path)

        dest_images_path = dest_path + 'images/'
        create_empty_folder(dest_images_path)

        dest_masks_path = dest_path + 'masks/'
        create_empty_folder(dest_masks_path)

        for image in Path(images_path).glob('*'):

            image_name = os.path.basename(image)
            shutil.copy(image, dest_images_path + image_name)
            corresponding_mask = masks_path + image_name
            shutil.copy(corresponding_mask, dest_masks_path + image_name)
    
    def _apply_transform(self, image, mask, tr):
        if tr == 'hf':
            transform = A.Compose([
            A.HorizontalFlip(p=0.5),
        ])
        elif tr == 'vf':
            transform = A.Compose([
            A.VerticalFlip(p=1),
        ])
        elif tr == 'rr':
            transform = A.Compose([
            A.SafeRotate(limit=25, p=1),
        ])
        elif tr == 'zc':
            transform = A.Compose([
            A.CropAndPad(percent=-0.1, p=1),
        ])

        transformed = transform(image=image, mask=mask)
        return transformed

    def augment(self, save_path:str):
        '''
        Applies augmentations using albumentations. The following augmentations
        are applied:
        1. Horizontal Flip (p=0.5)
        2. Vertical Flip (p=0.5)
        3. Random rotation (+25 to -25 degrees)
        4. Zooming/Cropping (within 80%, p=1)
        Each image may have 4 corresponding augmented images and masks. 
        Validation dataset will only contain original images + masks
        '''
        create_empty_folder(save_path)

        images_aug_path = save_path + 'images/'
        create_empty_folder(images_aug_path)

        masks_aug_path = save_path + 'masks/'
        create_empty_folder(masks_aug_path)

        for i, (image_path, mask_path) in enumerate(zip(self.image_paths, self.mask_paths)):
            # Read image and mask
            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            #Horizontal Flip
            try:
                hf = self._apply_transform(image, mask, 'hf')
                cv2.imwrite(images_aug_path + os.path.basename(image_path)[:-4] 
                            + '_hf.jpg', hf['image'])
                cv2.imwrite(masks_aug_path + os.path.basename(image_path)[:-4] 
                            + '_hf.jpg', hf['mask'])
            except Exception:
                pass
            #Vertical Flip
            try:
                vf = self._apply_transform(image, mask, 'vf')
                cv2.imwrite(images_aug_path + os.path.basename(image_path)[:-4] 
                            + '_vf.jpg', vf['image'])
                cv2.imwrite(masks_aug_path + os.path.basename(image_path)[:-4] 
                            + '_vf.jpg', vf['mask'])

            except Exception:
                pass
            #Rotate
            try:
                rr = self._apply_transform(image, mask, 'rr')
                cv2.imwrite(images_aug_path + os.path.basename(image_path)[:-4] 
                            + '_rr.jpg', rr['image'])
                cv2.imwrite(masks_aug_path + os.path.basename(image_path)[:-4] 
                            + '_rr.jpg', rr['mask'])
            except Exception:
                pass
            #Zoom/Crop
            try:
                zc = self._apply_transform(image, mask, 'zc')
                cv2.imwrite(images_aug_path + os.path.basename(image_path)[:-4] 
                            + '_zc.jpg', zc['image'])
                cv2.imwrite(masks_aug_path + os.path.basename(image_path)[:-4] 
                            + '_zc.jpg', zc['mask'])
            except Exception:
                pass

class WSegDataset(WoundDataset):
    '''
    Subclass of WoundDataset for WSeg specific functions
    '''
    def __init__(self, home_path: str):
        super().__init__(home_path)
        self.padded_images_path = self.home_path + 'padded_images/'
        create_empty_folder(self.padded_images_path)

        self.padded_masks_path = self.home_path + 'padded_masks/'
        create_empty_folder(self.padded_masks_path)

    def process_wseg(self):
        '''
        Processing the images and masks of wseg data. The images and masks are 
        already cropped. This function adds padding and copies them into 'padded_images'
        and 'padded_masks' folder inside the same home_path
        '''
        self.padding_wseg(im_path=self.images_path, save_path=self.padded_images_path)

        self.padding_wseg(im_path=self.masks_path, save_path=self.padded_masks_path, mask = True)
    
    def padding_wseg(self, im_path, save_path = None, mask = False):
        '''
        Adds padding to images in im_path and stores them in save_path
        '''
        image_paths = get_list_of_paths(im_path)
        image_names = get_list_of_file_names(im_path)

        for i, image_path in enumerate(image_paths):
            img = cv2.imread(str(image_path))
            img_height, img_width, _ = img.shape
            if mask: 
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                img_height, img_width = img.shape
            aspect_ratio = img_width/img_height

            if img_width > img_height:
                new_width = 224
                new_height = int(224 / aspect_ratio) 
            else:
                new_height = 224
                new_width = int(224 * aspect_ratio)

            img2 = cv2.resize(img.copy(), (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            if img_height > 224 or img_width > 224:
                img2 = cv2.resize(img.copy(), (new_width, new_height), interpolation=cv2.INTER_AREA)

            if new_height < 224:
                padding_bottom = 224 - new_height
                padded_image = cv2.copyMakeBorder(img2.copy(), 0, padding_bottom, 0, 0, 
                                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))

                
            if new_width < 224:
                padding_right = 224 - new_width
                padded_image = cv2.copyMakeBorder(img2.copy(), 0, 0, 0, padding_right, 
                                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))

            cv2.imwrite(save_path + image_names[i], padded_image)


def create_processed_dir(data_path: str, raw_dataset: WoundDataset, augment_dataset: WoundDataset):
    total_images = raw_dataset.len + augment_dataset.len
    val_length = round(total_images*0.1)
    test_length = round(total_images*0.05)

    val_names = random.sample(raw_dataset.image_names, val_length)
    test_names = random.sample(set(raw_dataset.image_names)-set(val_names), test_length)
    train_names = list(set(raw_dataset.image_names) - set(val_names)-set(test_names))

    create_empty_folder(data_path + 'processed/train/images/')
    create_empty_folder(data_path + 'processed/train/masks/')
    create_empty_folder(data_path + 'processed/val/images/')
    create_empty_folder(data_path + 'processed/val/masks/')
    create_empty_folder(data_path + 'processed/test/images/')
    create_empty_folder(data_path + 'processed/test/masks/')

    raw_dataset.move(dest_path = data_path + 'processed/train/')
    augment_dataset.move(dest_path = data_path + 'processed/train/')

    train_dataset = WoundDataset(home_path=data_path + 'processed/train/')

    
    for image_name in val_names:
        index = train_dataset.image_names.index(image_name)
        shutil.move(str(train_dataset.image_paths[index]), data_path + 'processed/val/images/')
        shutil.move(str(train_dataset.mask_paths[index]), data_path + 'processed/val/masks/')
    
    for image_name in test_names:
        index = train_dataset.image_names.index(image_name)
        shutil.move(str(train_dataset.image_paths[index]), data_path + 'processed/test/images/')
        shutil.move(str(train_dataset.mask_paths[index]), data_path + 'processed/test/masks/')
    
    train_dataset = WoundDataset(home_path=data_path + 'processed/train/')
    val_dataset = WoundDataset(home_path=data_path + 'processed/val/')
    return train_dataset, val_dataset


def make_dataset1(data_path:str):
    '''
    Process images and masks form dataset1 from external until interim
    Inside the external data folder:
    Dataset1 - Wound Segmentation Dataset
    Input:
        data_path - path to where the datasets are stored
    '''

    # Move cropped + padded images and masks to raw
    external_dataset = WoundDataset(data_path + 'external/dataset1/')
    external_dataset.move(data_path + 'raw/dataset1/')

    # Create augmentations
    raw_dataset = WoundDataset(data_path + 'raw/dataset1/')
    raw_dataset.augment(save_path = data_path + 'interim/dataset1/')

    # Create train and validation datasets
    raw_dataset = WoundDataset(data_path + 'raw/dataset1/')
    augment_dataset = WoundDataset(data_path + 'interim/dataset1/')

    _, _ = create_processed_dir(data_path=data_path,
                                                      raw_dataset=raw_dataset, 
                                                      augment_dataset=augment_dataset)


def make_dataset2(data_path:str):
    '''
    Process images and masks form dataset2 from external until interim
    Inside the external data folder:
    Dataset2 - WSeg Dataset (images are already cropped)
    Input:
        data_path - path to where the datasets are stored
    '''

    # Create padded images and masks and move to 'raw'
    external_dataset = WSegDataset(data_path + 'external/dataset2/')

    external_dataset.process_wseg()

    external_dataset.move(dest_path=data_path+'raw/dataset2/', 
                          images_path=external_dataset.padded_images_path, 
                          masks_path=external_dataset.padded_masks_path)
    
    # Create augmentations
    raw_dataset = WSegDataset(data_path + 'raw/dataset2/')

    raw_dataset.augment(save_path = data_path + 'interim/dataset2/')

    # Create train and validation datasets
    raw_dataset = WoundDataset(data_path + 'raw/dataset2/')
    augment_dataset = WoundDataset(data_path + 'interim/dataset2/')

    _, _ = create_processed_dir(data_path=data_path,
                                                      raw_dataset=raw_dataset, 
                                                      augment_dataset=augment_dataset)
    

if __name__=='__main__':
    pass
