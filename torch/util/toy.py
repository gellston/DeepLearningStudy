import torch
import torch.nn as nn


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_filters, out_filters, dilation=1, padding='same'):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_filters, out_channels=in_filters, kernel_size=3, padding=padding, groups=in_filters, dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class residual_separable_conv_relu(nn.Module):
    def __init__(self, in_filters, out_filters, dilation=1, padding='same'):
        super(residual_separable_conv_relu, self).__init__()
        self.separable_conv = depthwise_separable_conv(in_filters=in_filters, out_filters=out_filters, dilation=dilation, padding=padding)
        self.identity = nn.Identity()
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.separable_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        shortcut = self.identity(x)
        x = x + shortcut
        return x