import torch
import torch.nn.functional as F
import torch.nn as nn


class depthwise_separable_convV2(nn.Module):
    def __init__(self, in_filters, out_filters, dilation=1, padding='same'):
        super(depthwise_separable_convV2, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_filters,
                                   out_channels=in_filters,
                                   kernel_size=3,
                                   padding=padding,
                                   groups=in_filters,
                                   dilation=dilation,
                                   bias=False)
        self.relu6 = nn.ReLU6()
        self.bn1 = nn.BatchNorm2d(in_filters)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.pointwise = nn.Conv2d(in_channels=in_filters,
                                   out_channels=out_filters,
                                   kernel_size=1,
                                   bias=False)

    def forward(self, x):
        ## Depth wise convolution
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu6(x)

        ## Point wise convolution
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu6(x)
        return x


class bottleneck_residual_block(nn.Module):
    def __init__(self, in_filters=32, stride=1, expand_ratio=6, padding='same', dilation=1):
        super(bottleneck_residual_block, self).__init__()

        self.out_filters = in_filters * expand_ratio
        self.stride = stride

        ## Exapnd Layer
        self.exapnd_layer = nn.Conv2d(in_channels=in_filters,
                                      out_channels=self.out_filters,
                                      kernel_size=1,
                                      padding=padding,
                                      dilation=dilation,
                                      bias=False)

        self.batch_norm1_1 = nn.BatchNorm2d(self.out_filters)
        self.batch_norm1_2 = nn.BatchNorm2d(self.out_filters)
        self.batch_norm2 = nn.BatchNorm2d(in_filters)
        self.relu6 = nn.ReLU6()

        ## Depth wise Layer
        self.depthwise = nn.Conv2d(in_channels=self.out_filters,
                                   out_channels=self.out_filters,
                                   kernel_size=3,
                                   stride=self.stride,
                                   padding=padding,
                                   groups=self.out_filters,
                                   dilation=dilation,
                                   bias=False)

        ## Projection Layer
        self.projection = nn.Conv2d(in_channels=self.out_filters,
                                    out_channels=in_filters,
                                    kernel_size=1,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=False)



        self.identity = nn.Identity()

    def forward(self, x):

        ##Exapnd layer
        x = self.exapnd_layer(x)
        x = self.batch_norm1_1(x)
        x = self.relu6(x)
        ##Depth wise layer
        x = self.depthwise(x)
        x = self.batch_norm1_2(x)
        x = self.relu6(x)
        ##Projection layer
        x = self.projection(x)
        x = self.batch_norm2(x)
        ##identity Layer
        if self.stride == 1:
            shortcut = self.identity(x)
            x = x + shortcut

        return x


class CustomSegmentationV1(torch.nn.Module):

    def __init__(self):
        super(CustomSegmentationV1, self).__init__()
        #256x256
        self.layer1 = nn.Conv2d(in_channels=3,
                                out_channels=32,
                                dilation=1,
                                bias=False,
                                kernel_size=3,
                                stride=1)
        self.bn_layer1 = nn.BatchNorm2d(32)
        self.layer1_bottleneck = bottleneck_residual_block(in_filters=32,
                                                           expand_ratio=1)


        # 128x128
        self.layer2 = nn.Conv2d(in_channels=32,
                                out_channels=64,
                                dilation=1,
                                bias=False,
                                kernel_size=3,
                                stride=2)
        self.bn_layer2 = nn.BatchNorm2d(64)
        self.layer2_bottleneck = bottleneck_residual_block(in_filters=64)


        # 64x64
        self.layer3 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                dilation=1,
                                bias=False,
                                kernel_size=3,
                                stride=2)
        self.bn_layer3 = nn.BatchNorm2d(128)
        self.layer3_bottleneck = bottleneck_residual_block(in_filters=128)



        # 32x32
        self.layer4 = nn.Conv2d(in_channels=128,
                                out_channels=256,
                                dilation=1,
                                bias=False,
                                kernel_size=3,
                                stride=2)
        self.bn_layer4 = nn.BatchNorm2d(256)
        self.laye4_bottleneck = bottleneck_residual_block(in_filters=256)

        # 16x16
        self.layer4 = nn.Conv2d(in_channels=256,
                                out_channels=512,
                                dilation=1,
                                bias=False,
                                kernel_size=3,
                                stride=2)
        self.bn_layer4 = nn.BatchNorm2d(512)
        self.laye4_bottleneck = bottleneck_residual_block(in_filters=512)


    def forward(self, x):


        return x