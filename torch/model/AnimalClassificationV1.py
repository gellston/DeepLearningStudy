import numpy as np
import torch
import torch.nn as nn
import util.toy as toy
#import torch.nn.functional as F

class AnimalClassificationV1(nn.Module):
    ## output feature size
    ## O = (I - K + 2P) / S + 1
    def __init__(self):
        super(AnimalClassificationV1, self).__init__()
        self.sepConv1 = toy.depthwise_separable_conv(in_filters=3, out_filters=32, dilation=1, padding='same')
        self.sepConv2 = toy.depthwise_separable_conv(in_filters=32, out_filters=64, dilation=1, padding='same')
        self.sepConv3 = toy.depthwise_separable_conv(in_filters=64, out_filters=128, dilation=1, padding='same')
        self.sepConv4 = toy.depthwise_separable_conv(in_filters=128, out_filters=256, dilation=1, padding='same')

        self.max_pool = nn.MaxPool2d(kernel_size=2, padding=1, stride=2)
        self.residual_block1 = toy.residual_separable_conv_relu(in_filters=32, out_filters=32, dilation=1)
        self.residual_block2 = toy.residual_separable_conv_relu(in_filters=64, out_filters=64, dilation=1)
        self.residual_block3 = toy.residual_separable_conv_relu(in_filters=128, out_filters=128, dilation=1)
        self.residual_block4 = toy.residual_separable_conv_relu(in_filters=256, out_filters=256, dilation=1)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)


        self.final_conv = nn.Conv2d(in_channels=256, out_channels=5, kernel_size=3, stride=1, padding='same', dilation=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=14)


    def forward(self, x):

        x = x.permute(0, 3, 1, 2)

        ## Block1
        x = self.sepConv1(x) #100x100x32
        x = self.batch_norm1(x) #100x100x32
        x = self.residual_block1(x) #100x100x32
        #x = self.residual_block1(x)  # 100x100x32
        x = self.max_pool(x) #50x50x32

        ## Block2
        x = self.sepConv2(x) #50x50x64
        x = self.batch_norm2(x) #50x50x64
        x = self.residual_block2(x) #50x50x64
        #x = self.residual_block2(x)  # 50x50x64
        x = self.max_pool(x)  #25x25x64

        ## Block3
        x = self.sepConv3(x) #12x12x128
        x = self.batch_norm3(x) #12x12x128
        x = self.residual_block3(x)  #12x12x128
        #x = self.residual_block3(x)  # 12x12x128
        x = self.max_pool(x)  # 6x6x128


        ## Block3
        x = self.sepConv4(x) #12x12x256
        x = self.batch_norm4(x) #12x12x256
        x = self.residual_block4(x)  #12x12x256
        #x = self.max_pool(x)  # 6x6x256

        ## Global average pooling
        x = self.final_conv(x)
        x = self.avg_pool(x)
        x = torch.softmax(x, dim=1)

        return x
