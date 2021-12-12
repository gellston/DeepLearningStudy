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
                                   kernel_size=1, bias=False)

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



class separable_conv_residual(nn.Module):
    def __init__(self, in_filters=3, out_filters=3, dilation=1, padding=1):
        super(separable_conv_residual, self).__init__()
        self.separable_conv = depthwise_separable_convV2(in_filters=in_filters,
                                                         out_filters=out_filters,
                                                         dilation=dilation,
                                                         padding=padding)
        self.identity = nn.Identity()
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU6()

    def forward(self, x):
        x = self.separable_conv(x)
        x = self.bn(x)
        skip = self.identity(x)
        summation = skip + x
        x = self.relu(summation)
        return x



class down(nn.Module):
    def __init__(self, in_filters=3, out_filters=32, stride=1, padding=1):
        super(down, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_filters,
                              out_channels=out_filters,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              kernel_size=3)
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU6()

        self.separable_residual = separable_conv_residual(in_filters=out_filters,
                                                          out_filters=out_filters)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.separable_residual(x)
        return x

class up(nn.Module):
    def __init__(self, in_filters=3 , out_filters=3, stride=1, padding=1):
        super(up, self).__init__()
        self.transpose_conv = nn.ConvTranspose2d(in_channels=in_filters,
                                                 out_channels=out_filters,
                                                 stride=stride,
                                                 padding=padding,
                                                 kernel_size=2)
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU6()

        self.separable_residual = separable_conv_residual(in_filters=out_filters,
                                                          out_filters=out_filters)


    def forward(self, x):
        x = self.transpose_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.separable_residual(x)
        return x


class CustomSegmentationV1(torch.nn.Module):

    def __init__(self):
        super(CustomSegmentationV1, self).__init__()
        #256x256
        self.down1 = down(in_filters=3, out_filters=32, stride=1, padding=1)

        #128x128
        self.down2 = down(in_filters=32, out_filters=64, stride=2, padding=1)

        #64x64
        self.down3 = down(in_filters=64, out_filters=128, stride=2, padding=1)

        #32x32
        self.down4 = down(in_filters=128, out_filters=256, stride=2, padding=1)

        #16x16 (center)
        self.center = down(in_filters=256, out_filters=256, stride=2, padding=1)

        #32x32
        self.up4 = up(in_filters=256, out_filters=256, stride=2, padding=0)

        #64x64
        self.up3 = up(in_filters=256, out_filters=128, stride=2, padding=0)

        #128x128
        self.up2 = up(in_filters=128, out_filters=64, stride=2, padding=0)

        #256x256
        self.up1 = up(in_filters=64, out_filters=32, stride=2, padding=0)
        self.final = nn.Conv2d(in_channels=32, out_channels=1, padding=1, kernel_size=3, dilation=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        center = self.center(down4)

        up4 = self.up4(center)
        sum4 = up4 + down4
        up3 = self.up3(sum4)
        sum3 = up3 + down3
        up2 = self.up2(sum3)
        sum2 = up2 + down2
        up1 = self.up1(sum2)
        sum1 = up1 + down1
        final = self.final(sum1)
        y = self.sigmoid(final)
        return y