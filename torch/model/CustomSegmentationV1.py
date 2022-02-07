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

class Down(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=32, stride=1, expand_ratio=1):
        super(Down, self).__init__()
        self.layer = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               dilation=1,
                               bias=False,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.bottleneck = bottleneck_residual_block(in_filters=out_channel,
                                                    expand_ratio=expand_ratio,
                                                    padding='same',
                                                    dilation=1)
        self.relu = nn.ReLU6()

    def forward(self, x):
        x = self.layer(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.bottleneck(x)
        x = self.relu(x)
        return x

class Up(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=32, stride=2, expand_ratio=1):
        super(Up, self).__init__()
        self.layer = nn.ConvTranspose2d(in_channels=in_channel,
                                        out_channels=out_channel,
                                        dilation=1,
                                        bias=False,
                                        kernel_size=2,
                                        stride=stride)
        self.bn = nn.BatchNorm2d(out_channel)
        self.bottleneck = bottleneck_residual_block(in_filters=out_channel,
                                                    expand_ratio=expand_ratio)
        self.relu = nn.ReLU6()

    def forward(self, x):
        x = self.layer(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.bottleneck(x)
        x = self.relu(x)

        return x


class CustomSegmentationV1(torch.nn.Module):

    def __init__(self):
        super(CustomSegmentationV1, self).__init__()


        self.exapand_rate = 2

        #256x256
        self.down1 = Down(in_channel=3, out_channel=32, stride=1, expand_ratio=1)

        # 128x128
        self.down2 = Down(in_channel=32, out_channel=64, stride=2, expand_ratio=self.exapand_rate)

        # 64x64
        self.down3 = Down(in_channel=64, out_channel=128, stride=2, expand_ratio=self.exapand_rate)

        # 32x32
        self.down4 = Down(in_channel=128, out_channel=256, stride=2, expand_ratio=self.exapand_rate)

        # 16x16
        self.down5 = Down(in_channel=256, out_channel=256, stride=2, expand_ratio=self.exapand_rate)
        self.center1 = bottleneck_residual_block(in_filters=256, stride=1, expand_ratio=self.exapand_rate)
        #self.center2 = bottleneck_residual_block(in_filters=256, stride=1, expand_ratio=self.exapand_rate)

        # 32x32
        self.up4 = Up(in_channel=256, out_channel=256, stride=2, expand_ratio=self.exapand_rate)

        # 64x64
        self.up3 = Up(in_channel=256, out_channel=128, stride=2, expand_ratio=self.exapand_rate)

        # 128x128
        self.up2 = Up(in_channel=128, out_channel=64, stride=2, expand_ratio=self.exapand_rate)

        # 256x256
        self.up1 = Up(in_channel=64, out_channel=32, stride=2, expand_ratio=self.exapand_rate)
        #self.final1 = bottleneck_residual_block(in_filters=32, stride=1, expand_ratio=self.exapand_rate)
        self.final1 = nn.Conv2d(in_channels=32,
                                out_channels=1,
                                dilation=1,
                                bias=False,
                                kernel_size=3,
                                padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        center1 = self.down5(down4)
        center2 = self.center1(center1)
        #center3 = self.center2(center2)

        up4 = self.up4(center2)
        sum4 = torch.add(up4, down4)

        up3 = self.up3(sum4)
        sum3 = torch.add(up3, down3)

        up2 = self.up2(sum3)
        sum2 = torch.add(up2, down2)

        up1 = self.up1(sum2)
        sum1 = torch.add(up1, down1)
        #final1 = self.final1(sum1)
        final = self.final1(sum1)
        y = self.sigmoid(final)

        return y