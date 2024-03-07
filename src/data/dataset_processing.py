import os
#import glob
import shutil
import random
from pathlib import Path
import cv2
import albumentations as A
from tqdm import tqdm
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

        num_files = sum(1 for _ in Path(images_path).glob('*'))  
        for image in tqdm(Path(images_path).glob('*'), total=num_files):

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
        elif tr == 'od':
            transform = A.Compose([
            A.OpticalDistortion(),
        ])
        elif tr == 'gd':
            transform = A.Compose([
            A.GridDistortion(),
        ])
        elif tr == 'br':
            transform = A.Compose([
            A.Blur(),
        ])
        elif tr == 'rbc':
            transform = A.Compose([
            A.RandomBrightnessContrast(),
        ])
        elif tr == 'tr':
            transform = A.Compose([
            A.Transpose(),
        ])
        transformed = transform(image=image, mask=mask)
        return transformed
    
    def _save_transform(self, image, mask, tr, image_path):
        '''
        Save the transform
        '''
        transformed = self._apply_transform(image, mask, tr)
        cv2.imwrite(self.images_aug_path + os.path.basename(image_path)[:-4] 
                    + f'_{tr}.jpg', transformed['image'])
        cv2.imwrite(self.masks_aug_path + os.path.basename(image_path)[:-4] 
                    + f'_{tr}.jpg', transformed['mask'])


    def augment(self, save_path:str, ws_aug:bool):
        '''
        Applies augmentations using albumentations. The following augmentations
        are applied for MobileNetV2 implementation:
        1. Horizontal Flip (p=0.5)
        2. Vertical Flip (p=0.5)
        3. Random rotation (+25 to -25 degrees)
        4. Zooming/Cropping (within 80%, p=1)

        The following are applied for WSeg:
        1. Horizontal Flip (p=0.5)
        2. Random rotation (+25 to -25 degrees)
        3. Optical Distortion (default values)
        4. Grid Distortion (default values)
        5. Blur (default values)
        6. Random Brightness (default values)
        7. Transpose

        Each image may have 4 (or 7 for WSeg) corresponding augmented images and masks. 
        Validation dataset will only contain original images + masks
        '''
        create_empty_folder(save_path)

        self.images_aug_path = save_path + 'images/'
        create_empty_folder(self.images_aug_path)

        self.masks_aug_path = save_path + 'masks/'
        create_empty_folder(self.masks_aug_path)

        for i, (image_path, mask_path) in tqdm(enumerate(zip(self.image_paths, self.mask_paths)), total=len(self.image_paths)):
            # Read image and mask
            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Common augmentations
            #Horizontal Flip
            try:
                tr = 'hf'
                self._save_transform(image, mask, tr, image_path)
            except Exception:
                pass
            #Rotate
            try:
                tr = 'rr'
                self._save_transform(image, mask, tr, image_path)
            except Exception:
                pass

            # MobileNetV2 only augmentations
            if not ws_aug:
                #Vertical Flip
                try:
                    tr = 'vf'
                    self._save_transform(image, mask, tr, image_path)
                except Exception:
                    pass
                #Zoom/Crop
                try:
                    tr = 'zc'
                    self._save_transform(image, mask, tr, image_path)
                except Exception:
                    pass
            
            #WSeg only augmentations
            if ws_aug:
                #Optical Distortion
                try:
                    tr = 'od'
                    self._save_transform(image, mask, tr, image_path)
                except Exception:
                    pass
                #Grid Distortion
                try:
                    tr = 'gd'
                    self._save_transform(image, mask, tr, image_path)
                except Exception:
                    pass
                #Blur
                try:
                    tr = 'br'
                    self._save_transform(image, mask, tr, image_path)                    
                except Exception:
                    pass
                #Random Brightness Contrast
                try:
                    tr = 'rbc'
                    self._save_transform(image, mask, tr, image_path)                    
                except Exception:
                    pass
                #Transpose
                try:
                    tr = 'tr'
                    self._save_transform(image, mask, tr, image_path)                    
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
