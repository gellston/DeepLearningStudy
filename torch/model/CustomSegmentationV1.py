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


class convolution_bn_relu6(nn.Module):
    def __init__(self, in_filters=32, out_filters=32, stride=1):
        self.conv = nn.Conv2d(in_channels=in_filters,
                              out_channels=out_filters,
                              dilation=1,
                              bias=False,
                              kernel_size=3,
                              stride=stride)
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu6 = nn.ReLU6()


    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)

        return x



class CustomSegmentationV1(torch.nn.Module):

    def __init__(self):
        super(CustomSegmentationV1, self).__init__()
        #256x256
        self.layer_down1 = convolution_bn_relu6(in_filters=3, out_filters=8, stride=1)
        self.layer_down1_bottleneck1 = bottleneck_residual_block(in_filters=8, expand_ratio=1)
        self.layer_down1_bottleneck2 = bottleneck_residual_block(in_filters=8, expand_ratio=1)


        # 128x128
        self.layer_down2 = convolution_bn_relu6(in_filters=8, out_filters=32, stride=2)
        self.layer_down2_bottleneck1 = bottleneck_residual_block(in_filters=32)
        self.layer_down2_bottleneck2 = bottleneck_residual_block(in_filters=32)


        # 64x64
        self.layer_down3 = convolution_bn_relu6(in_filters=32, out_filters=64, stride=2)
        self.layer_down3_bottleneck1 = bottleneck_residual_block(in_filters=64)
        self.layer_down3_bottleneck2 = bottleneck_residual_block(in_filters=64)



        # 32x32
        self.layer_down4 = convolution_bn_relu6(in_filters=64, out_filters=128, stride=2)
        self.layer_down4_bottlenect1 = bottleneck_residual_block(in_filters=128)
        self.layer_down4_bottlenect2 = bottleneck_residual_block(in_filters=128)


        # 16x16 ### Center
        self.layer_down5 = convolution_bn_relu6(in_filters=128, out_filters=128, stride=2)
        self.layer_down5_bottlenect1 = bottleneck_residual_block(in_filters=128)
        self.layer_down5_bottlenect2 = bottleneck_residual_block(in_filters=128)
        self.layer_down5_bottlenect3 = bottleneck_residual_block(in_filters=128)





        # deconv 32
        self.layer_up4 = nn.ConvTranspose2d(kernel_size=3,
                                            bias=False,
                                            in_channels=128,
                                            out_channels=128,
                                            stride=2)


    def forward(self, x):


        return x