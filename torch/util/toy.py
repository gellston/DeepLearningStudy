import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class depthwise_separable_convV2(nn.Module):
    def __init__(self, in_filters, out_filters, dilation=1, padding='same'):
        super(depthwise_separable_convV2, self).__init__()

        self.depthwise_conv = nn.Sequential(nn.Conv2d(in_channels=in_filters,
                                                      out_channels=in_filters,
                                                      kernel_size=3,
                                                      padding=padding,
                                                      groups=in_filters,
                                                      dilation=dilation,
                                                      bias=False),
                                            nn.BatchNorm2d(in_filters),
                                            nn.ReLU())
        self.pointwise_conv = nn.Sequential(nn.Conv2d(in_channels=in_filters,
                                                      out_channels=out_filters,
                                                      kernel_size=1,
                                                      bias=False),
                                            nn.BatchNorm2d(out_filters),
                                            nn.ReLU())


    def forward(self, x):
        ## Depth wise convolution
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
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
        x = self.relu6(x)
        ##identity Layer
        if self.stride == 1:
            shortcut = self.identity(x)
            x = x + shortcut

        return x






class depthwise_separable_conv(nn.Module):
    def __init__(self, in_filters, out_filters, dilation=1, padding='same'):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_filters, out_channels=in_filters, kernel_size=3, padding=padding, groups=in_filters, dilation=dilation, bias=False)
        self.pointwise = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class residual_separable_conv_relu(nn.Module):
    def __init__(self, in_filters, out_filters, dilation=1, padding='same'):
        super(residual_separable_conv_relu, self).__init__()
        self.separable_conv = depthwise_separable_conv(in_filters=in_filters, out_filters=out_filters, dilation=dilation, padding=padding, bias=False)
        self.identity = nn.Identity()
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.identity(x)
        x = self.separable_conv(x)
        x = self.batch_norm(x)
        x = x + shortcut
        x = self.activation(x)
        return x


class residual_conv_relu(nn.Module):
    def __init__(self, in_filters, out_filters, dilation=1, padding='same'):
        super(residual_conv_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=3, dilation=dilation, padding=padding, bias=False)
        self.identity = nn.Identity()
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.identity(x)

        x = self.conv(x)
        x = self.batch_norm(x)
        x = x + shortcut
        x = self.activation(x)
        return x