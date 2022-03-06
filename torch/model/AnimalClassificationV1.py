import numpy as np
import torch
import torch.nn as nn
import util.toy as toy
import torch.nn.functional as F

class AnimalClassificationV1(nn.Module):
    ## output feature size
    ## O = (I - K + 2P) / S + 1
    def __init__(self, num_class=5):
        super(AnimalClassificationV1, self).__init__()

        self.num_class = num_class

        self.layer1_conv = nn.Sequential(
            toy.depthwise_separable_conv(in_filters=3, out_filters=32, dilation=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            toy.residual_conv_relu(in_filters=32, out_filters=32, dilation=1),
            toy.residual_conv_relu(in_filters=32, out_filters=32, dilation=1),
            nn.MaxPool2d(kernel_size=2, padding=1, stride=2))

        self.layer2_conv = nn.Sequential(
            toy.depthwise_separable_conv(in_filters=32, out_filters=64, dilation=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            toy.residual_conv_relu(in_filters=64, out_filters=64, dilation=1),
            toy.residual_conv_relu(in_filters=64, out_filters=64, dilation=1),
            nn.MaxPool2d(kernel_size=2, padding=1, stride=2))

        self.layer3_conv = nn.Sequential(
            toy.depthwise_separable_conv(in_filters=64, out_filters=128, dilation=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            toy.residual_conv_relu(in_filters=128, out_filters=128, dilation=1),
            toy.residual_conv_relu(in_filters=128, out_filters=128, dilation=1),
            nn.MaxPool2d(kernel_size=2, padding=1, stride=2))

        self.layer4_conv = nn.Sequential(
            toy.depthwise_separable_conv(in_filters=128, out_filters=256, dilation=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            toy.residual_conv_relu(in_filters=256, out_filters=256, dilation=1),
            toy.residual_conv_relu(in_filters=256, out_filters=256, dilation=1),
            nn.MaxPool2d(kernel_size=2, padding=1, stride=2))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=self.num_class,
                      kernel_size=1,
                      stride=1,
                      padding='same',
                      dilation=1))

        self.avg_pool = nn.Sequential(torch.nn.AdaptiveAvgPool2d(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.layer1_conv(x)
        x = self.layer2_conv(x)
        x = self.layer3_conv(x)
        x = self.layer4_conv(x)

        ## Global average pooling
        x = self.final_conv(x)
        x = self.avg_pool(x)
        x = x.view([-1, self.num_class])
        x = self.sigmoid(x)

        return x
