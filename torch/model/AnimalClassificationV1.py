import numpy as np
import torch
import torch.nn as nn
import util.toy as toy
import torch.nn.functional as F

class AnimalClassificationV1(nn.Module):
    ## output feature size
    ## O = (I - K + 2P) / S + 1
    def __init__(self):
        super(AnimalClassificationV1, self).__init__()

        self.sepConv1 = nn.Conv2d(in_channels=3, out_channels=32, dilation=1, kernel_size=3, padding='same', bias=False)
        self.sepConv2 = nn.Conv2d(in_channels=32, out_channels=64, dilation=1, kernel_size=3, padding='same', bias=False)
        self.sepConv3 = nn.Conv2d(in_channels=64, out_channels=128, dilation=1, kernel_size=3, padding='same', bias=False)
        self.sepConv4 = nn.Conv2d(in_channels=128, out_channels=256, dilation=1, kernel_size=3, padding='same', bias=False)
        self.sepConv5 = nn.Conv2d(in_channels=256, out_channels=512, dilation=1, kernel_size=3, padding='same', bias=False)

        torch.nn.init.xavier_uniform_(self.sepConv1.weight)
        torch.nn.init.xavier_uniform_(self.sepConv2.weight)
        torch.nn.init.xavier_uniform_(self.sepConv3.weight)
        torch.nn.init.xavier_uniform_(self.sepConv4.weight)
        torch.nn.init.xavier_uniform_(self.sepConv5.weight)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        self.max_pool = nn.MaxPool2d(kernel_size=2, padding=1, stride=2)
        self.swish = toy.Swish()

        self.residual_block1_1 = toy.residual_conv_relu(in_filters=32, out_filters=32, dilation=1)
        self.residual_block1_2 = toy.residual_conv_relu(in_filters=32, out_filters=32, dilation=1)


        self.residual_block2_1 = toy.residual_conv_relu(in_filters=64, out_filters=64, dilation=1)
        self.residual_block2_2 = toy.residual_conv_relu(in_filters=64, out_filters=64, dilation=1)


        self.residual_block3_1 = toy.residual_conv_relu(in_filters=128, out_filters=128, dilation=1)
        self.residual_block3_2 = toy.residual_conv_relu(in_filters=128, out_filters=128, dilation=1)
        self.residual_block3_3 = toy.residual_conv_relu(in_filters=128, out_filters=128, dilation=1)


        self.residual_block4_1 = toy.residual_conv_relu(in_filters=256, out_filters=256, dilation=1)
        self.residual_block4_2 = toy.residual_conv_relu(in_filters=256, out_filters=256, dilation=1)
        self.residual_block4_3 = toy.residual_conv_relu(in_filters=256, out_filters=256, dilation=1)


        self.residual_block5_1 = toy.residual_conv_relu(in_filters=512, out_filters=512, dilation=1)
        self.residual_block5_2 = toy.residual_conv_relu(in_filters=512, out_filters=512, dilation=1)
        self.residual_block5_3 = toy.residual_conv_relu(in_filters=512, out_filters=512, dilation=1)

        self.final_conv = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1, padding='same',
                                     dilation=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=14)



    def forward(self, x):

        ## Block1
        x = self.sepConv1(x) #100x100x32
        x = self.bn1(x)
        x = self.swish(x)
        x = self.residual_block1_1(x)
        x = self.residual_block1_2(x)
        x = self.max_pool(x) #50x50x32


        ## Block2
        x = self.sepConv2(x) #50x50x64
        x = self.bn2(x)
        x = self.swish(x)
        x = self.residual_block2_1(x)
        x = self.residual_block2_2(x)
        x = self.max_pool(x)  #25x25x64


        ## Block3
        x = self.sepConv3(x) #12x12x128
        x = self.bn3(x)
        x = self.swish(x)
        x = self.residual_block3_1(x)
        x = self.residual_block3_2(x)
        x = self.residual_block3_3(x)
        x = self.max_pool(x)  # 6x6x128


        ## Block4
        x = self.sepConv4(x) #12x12x256
        x = self.bn4(x)
        x = self.swish(x)
        x = self.residual_block4_1(x)
        x = self.residual_block4_2(x)
        x = self.residual_block4_3(x)
        #x = self.max_pool(x)



        ## Block5
        #x = self.sepConv5(x) #12x12x256
        #x = self.bn5(x)
        #x = self.swish(x)
        #x = self.residual_block5_1(x)
        #x = self.residual_block5_2(x)
        #x = self.residual_block5_3(x)





        ## Global average pooling
        x = self.final_conv(x)
        x = self.swish(x)
        x = self.avg_pool(x)
        x = x.view([-1, 2])
        x = F.softmax(x, dim=1)


        return x
