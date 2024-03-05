"""
Module defines the MobileNetV2 Encoder-Decoder architecture
"""

import torch
from torch import nn
import torchvision
import numpy as np

class MobileNetV2Decoder(nn.Module):
    """
    The class that defines the MobileNetV2 Decoder network from the paper.
    """
    def __init__(self, input_shape, classes):
        super(MobileNetV2Decoder, self).__init__()
        # Expected input_shape = [7, 7]

        #Sequence 1
        #Average Pooling 2D
        self.pool2d = nn.AvgPool2d(kernel_size=tuple(input_shape))

        #Conv2D, BatchNorm, ReLU
        self.conv2d1 = nn.Conv2d(in_channels=320, out_channels=256, kernel_size=(1,1), stride=(1,1), padding = 'same', bias=False)
        self.bn = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU()

        #BiLinear Upsampling
        self.biup1 = torchvision.transforms.Resize(tuple(input_shape))

        self.seq1 = nn.Sequential(self.pool2d, self.conv2d1, self.bn, self.relu, self.biup1)

        #Sequence 2
        #Conv2D, BatchNorm, ReLU
        self.seq2 = nn.Sequential(self.conv2d1, self.bn, self.relu)

        #Sequence 3
        #Conv2D, BatchNorm, ReLU
        self.conv2d2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1), stride=(1,1), padding = 'same', bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.conv2d3 = nn.Conv2d(in_channels=256, out_channels=classes, kernel_size=(1,1), stride=(1,1), padding = 'same', bias=False)
        self.biup2 = torchvision.transforms.Resize((224,224), antialias=True)

        #Adding sigmoid to the end for binary classification
        self.sig1 = nn.Sigmoid()

        self.seq3 = nn.Sequential(self.conv2d2, self.bn, self.relu, self.dropout, self.conv2d3, self.biup2, self.sig1)


    def forward(self, x):
        # ... implement the logic for your decoder's forward pass ...
        s1 = self.seq1(x)

        s2 = self.seq2(x)

        s3 = torch.cat((s1, s2), axis = 1)

        output = self.seq3(s3)

        return output


class MobileNetV2withDecoder(nn.Module):
    """
    The class that defines the MobileNetV2 Encoder-Decoder network.
    Args:
        stripped_model = MobileNetV2 without the classification layer and the final layer 
                            to fit the paper discription
    """
    def __init__(self, stripped_model: nn.Sequential, classes):
        super(MobileNetV2withDecoder, self).__init__()

        self.encoder = stripped_model

        self.decoder = MobileNetV2Decoder(input_shape=(7,7), classes=classes)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2