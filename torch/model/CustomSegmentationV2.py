import torch
import torch.nn as nn





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
        self.relu6_2 = nn.ReLU6()

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
        shortcut = self.identity(x)
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

        ##identity Layer
        if self.stride == 1:
            x = x + shortcut

        x = self.batch_norm2(x)
        x = self.relu6_2(x)

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


class CustomSegmentationV2(torch.nn.Module):

    def __init__(self):
        super(CustomSegmentationV2, self).__init__()


        self.exapand_rate = 3

        #200x56x32
        self.down1 = Down(in_channel=1, out_channel=16, stride=2, expand_ratio=1) #32

        # 100x28x64
        self.down2 = Down(in_channel=16, out_channel=32, stride=2, expand_ratio=self.exapand_rate) #64

        # 50x14x128
        self.down3 = Down(in_channel=32, out_channel=64, stride=2, expand_ratio=self.exapand_rate) #128

        # 25x7x256
        self.down4 = Down(in_channel=64, out_channel=128, stride=2, expand_ratio=self.exapand_rate) #256


        # 25x7x256
        self.center1 = bottleneck_residual_block(in_filters=128, stride=1, expand_ratio=self.exapand_rate) #256


        # 50x14x128
        self.up3 = Up(in_channel=128, out_channel=64, stride=2, expand_ratio=self.exapand_rate)  #128


        # 100x28x64
        self.up2 = Up(in_channel=64, out_channel=32, stride=2, expand_ratio=self.exapand_rate)   #64

        #200x56x32
        self.up1 = Up(in_channel=32, out_channel=16, stride=2, expand_ratio=self.exapand_rate)    #32

        self.final1 = nn.ConvTranspose2d(in_channels=16,
                                         out_channels=8,
                                         dilation=1,
                                         bias=False,
                                         kernel_size=2,
                                         stride=2)
        self.final1_bn = nn.BatchNorm2d(8)
        self.final1_relu6 = nn.ReLU6()

        self.final2 = nn.Conv2d(in_channels=8,
                                out_channels=1,
                                bias=False,
                                kernel_size=3,
                                padding='same')

        self.final2_bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        center = self.center1(down4)

        up3 = self.up3(center)
        sum3 = torch.add(up3, down3)

        up2 = self.up2(sum3)
        sum2 = torch.add(up2, down2)

        up1 = self.up1(sum2)
        sum1 = torch.add(up1, down1)

        final1 = self.final1(sum1)
        fianl1_batch_norm = self.final1_bn(final1)
        final1_relu6 = self.final1_relu6(fianl1_batch_norm)
        final2 = self.final2(final1_relu6)
        final2_batch_norm = self.final2_bn(final2)
        y = self.sigmoid(final2_batch_norm)

        return y