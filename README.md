# Wound Segmentation Reproduced

This repository is the reproduction 2 research papers:
1. [Fully automatic wound segmentation with deep convolutional neural networks](https://arxiv.org/abs/2010.05855)
2. [WSNet: Towards An Effective Method for Wound Image Segmentation](https://ieeexplore.ieee.org/document/10030591)

## Data Preprocessing
We pre-process data provided from both papers and train it on MobileNetV2. Both datasets go through a slightly different pipeline due to the inherent differences in the way the images are structured. During the process, the data goes through a series of folders:
```
1. external - Here the images and masks are raw and require some processing 
        before they can be moved to interim. Datasets are divided.
2. raw - Here the data is in the exact format we need them, image and mask.
        Datasets are divided.
3. interim - Here images and masks are places after augmentation. Datasets are divided.
4. processed - Here images and masks are places in their respective folder. 
            Datasets are combined, if both are used.
```
### Dataset 1
1. The images and masks are moved from external to raw folder.
2. The images and masks are divided into train, test and val and placed in raw folder.
3. Augmentations are applied on images and masks from the train folder, then everything is moved to processed folders.

### Dataset 2
1. Padding is applied to the images and masks are in external folder and then moved to raw folder.
2. The images and masks are divided into train, test and val and placed in raw folder.
3. Augmentations are applied on images and masks from the train folder, then everything is moved to processed folders.

#### Model 1
The model is an encoder - decoder style model where the encoder is MobileNetV2. The decoder is a custom decoder defined by the authors. Following is the visual structure:

#### Model 2
The classical LinkNet reproduction.

### Hyper-parameters and Loss functions
The hyper-parameters and the loss functions are hard coded into the config file according to what is written in the paper.

### Result
The results from the paper were partially reproducible.