import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple



def IOU(target, prediction):
    prediction = np.where(prediction > 0.5, 1, 0)
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class SeparableConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         groups=in_channels,
                                         bias=bias,
                                         stride=stride,
                                         padding=padding)

        self.pointwise = torch.nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=1,
                                         bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



class SeparableActivationConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=False, activation=torch.nn.SiLU):
        super(SeparableActivationConv2d, self).__init__()

        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         groups=in_channels,
                                         bias=bias,
                                         stride=stride,
                                         padding=padding)

        self.pointwise = torch.nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=1,
                                         bias=bias)

        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.activation1 = activation()

        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.activation2 = activation()


    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.activation2(out)
        return out


class ResidualBlock(torch.nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim, stride=1, activation=torch.nn.SiLU):
        super(ResidualBlock, self).__init__()

        self.stride = stride
        self.features = torch.nn.Sequential(torch.nn.Conv2d(in_dim,
                                                            mid_dim,
                                                            kernel_size=3,
                                                            stride=self.stride,
                                                            padding=1,
                                                            bias=False),
                                            torch.nn.BatchNorm2d(num_features=mid_dim),
                                            activation(),
                                            torch.nn.Conv2d(mid_dim,
                                                            out_dim,
                                                            kernel_size=3,
                                                            padding='same',
                                                            bias=False),
                                            torch.nn.BatchNorm2d(num_features=out_dim))

        self.down_skip_connection = torch.nn.Conv2d(in_channels=in_dim,
                                                    out_channels=out_dim,
                                                    kernel_size=1,
                                                    stride=self.stride)
        self.dim_equalizer = torch.nn.Conv2d(in_channels=in_dim,
                                             out_channels=out_dim,
                                             kernel_size=1)
        self.final_activation = activation()



    def forward(self, x):
        if self.stride == 2:
            down = self.down_skip_connection(x)
            out = self.features(x)
            out = out + down

        else:
            out = self.features(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        out = self.final_activation(out)
        return out





class DenseBottleNeck(torch.nn.Module):
    def __init__(self, in_channels, growth_rate=32, expansion_rate=4, droprate=0.2, activation=torch.nn.ReLU):
        super().__init__()

        inner_channels = expansion_rate * growth_rate           ##expansion_size=32*4
        self.droprate = droprate

        self.residual = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),                  ##ex:128
            activation(),
            torch.nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False),##32*4 #expansion layer
            torch.nn.BatchNorm2d(inner_channels),
            activation(),
            torch.nn.Conv2d(inner_channels, growth_rate, 3, stride=1, padding=1, bias=False) ##32
        )

        self.shortcut = torch.nn.Sequential()

    def forward(self, x):
        #output = F.dropout(self.residual(x), p=self.droprate, inplace=False, training=self.training)
        return torch.cat([self.shortcut(x), self.residual(x)], 1)



class ResidualBottleNeck(torch.nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, stride=1, activation=torch.nn.ReLU):
        super().__init__()

        self.stride = stride


        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=inner_channels,
                            kernel_size=1,
                            bias=False),
            torch.nn.BatchNorm2d(inner_channels),                  ##ex:128
            activation(),
            torch.nn.Conv2d(inner_channels,
                            inner_channels,
                            kernel_size=3,
                            stride=self.stride,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(inner_channels),
            activation(),
            torch.nn.Conv2d(inner_channels, out_channels, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),

        )

        self.final_activation = activation()
        self.down_skip_connection = torch.nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=1,
                                                    stride=self.stride)
        self.dim_equalizer = torch.nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1)

    def forward(self, x):
        if self.stride == 2:
            down = self.down_skip_connection(x)
            out = self.features(x)
            out = out + down

        else:
            out = self.features(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        out = self.final_activation(out)
        return out


class InvertedBottleNeck(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate=4, stride=1, activation=torch.nn.ReLU6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_rate = expansion_rate
        self.stride = stride

        self.expansion_out = int(self.in_channels * self.expansion_rate)

        self.conv_expansion = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1,
                                                                  in_channels=self.in_channels,
                                                                  out_channels=self.expansion_out,
                                                                  bias=False,
                                                                  stride=1),
                                                  torch.nn.BatchNorm2d(self.expansion_out),
                                                  activation())

        self.conv_depthwise = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=3,
                                                                  in_channels=self.expansion_out,
                                                                  out_channels=self.expansion_out,
                                                                  groups=self.expansion_out,
                                                                  bias=False,
                                                                  padding=1,
                                                                  stride=self.stride),
                                                  torch.nn.BatchNorm2d(self.expansion_out),
                                                  activation())

        self.conv_projection = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1,
                                                                   in_channels=self.expansion_out,
                                                                   out_channels=self.out_channels,
                                                                   bias=False),
                                                   torch.nn.BatchNorm2d(self.out_channels))


    def forward(self, x):
        out = self.conv_expansion(x)
        out = self.conv_depthwise(out)
        out = self.conv_projection(out)

        if self.stride != 2 and self.in_channels == self.out_channels:
            out = out + x

        return out


class Transition(torch.nn.Module):
    def __init__(self, in_channels, out_channels, droprate=0.2, activation=torch.nn.ReLU):
        super().__init__()
        self.droprate=droprate
        self.down_sample = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            activation(),
            torch.nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            torch.nn.AvgPool2d(2, stride=2)
        )
        self.spatial_dropout = torch.nn.Dropout2d(p=self.droprate)

    def forward(self, x):
        output = self.down_sample(x)
        return output


class DenseBlock(torch.nn.Module):

    def __init__(self, num_input_features, num_layers, expansion_rate, growth_rate, droprate, activation=torch.nn.ReLU):
        super(DenseBlock, self).__init__()

        self.dense_block = torch.nn.Sequential()
        self.droprate = droprate

        for i in range(num_layers):
            layer = DenseBottleNeck(in_channels=num_input_features + i * growth_rate,
                                    growth_rate=growth_rate,
                                    expansion_rate=expansion_rate,
                                    droprate=droprate,
                                    activation=activation)
            self.dense_block.add_module('denselayer_%d' % (i + 1), layer)
        self.spatial_dropout = torch.nn.Dropout2d(p=self.droprate)


    def forward(self, x):
        x = self.dense_block(x)
        x = self.spatial_dropout(x)
        #if self.droprate > 0:
            #x = F.dropout(x, p=self.droprate, inplace=False, training=self.training)
        return x


class CSPResidualBlock(torch.nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim, stride=1, part_ratio=0.5, activation=torch.nn.SiLU):
        super(CSPResidualBlock, self).__init__()

        self.part1_chnls = int(in_dim * part_ratio)
        self.part2_chnls = in_dim - self.part1_chnls                ##Residual Layer Channel Calculation

        self.part1_out_chnls = int(out_dim * part_ratio)
        self.part2_out_chnls = out_dim - self.part1_out_chnls

        self.stride = stride
        self.residual_block = torch.nn.Sequential(torch.nn.Conv2d(self.part2_chnls,
                                                                  mid_dim,
                                                                  kernel_size=3,
                                                                  stride=self.stride,
                                                                  padding=1,
                                                                  bias=False),
                                                  torch.nn.BatchNorm2d(num_features=mid_dim),
                                                  activation(),
                                                  torch.nn.Conv2d(mid_dim,
                                                                  self.part2_out_chnls,
                                                                  kernel_size=3,
                                                                  padding='same',
                                                                  bias=False),
                                                  torch.nn.BatchNorm2d(num_features=self.part2_out_chnls))

        self.projection1 = torch.nn.Conv2d(in_channels=self.part2_chnls,            ##Residual Projection
                                           out_channels=self.part2_out_chnls,
                                           kernel_size=1,
                                           stride=2)

        self.projection2 = torch.nn.Conv2d(in_channels=self.part1_chnls,
                                           out_channels=self.part1_out_chnls,
                                           kernel_size=1,
                                           stride=2)

        self.dim_equalizer1 = torch.nn.Conv2d(in_channels=self.part2_chnls,
                                              out_channels=self.part2_out_chnls,
                                              kernel_size=1)

        self.dim_equalizer2 = torch.nn.Conv2d(in_channels=self.part1_chnls,
                                              out_channels=self.part1_out_chnls,
                                              kernel_size=1)

        self.activation = activation()

    def forward(self, x):

        part1 = x[:, :self.part1_chnls, :, :] #part1 channel 자르기
        part2 = x[:, self.part1_chnls:, :, :] #part2 channel 자르기
        skip_connection = part2

        if self.stride == 2:
            skip_connection = self.projection1(skip_connection)
            part1 = self.projection2(part1)
        else:
            if self.part1_chnls != self.part1_out_chnls:
                skip_connection = self.dim_equalizer1(skip_connection)
                part1 = self.dim_equalizer2(part1)

        residual = self.residual_block(part2)  # F(x)
        residual = torch.add(residual, skip_connection)
        residual = self.activation(residual)
        out = torch.cat((part1, residual), 1)
        return out


class CSPDenseBlock(torch.nn.Module):

    def __init__(self, num_input_features, num_layers, expansion_rate, growth_rate, droprate, part_ratio=0.5, activation=torch.nn.ReLU):
        super(CSPDenseBlock, self).__init__()


        self.part1_chnls = int(num_input_features * part_ratio)
        self.part2_chnls = num_input_features - self.part1_chnls ##Dense Layer Channel Calculation


        self.dense_block = torch.nn.Sequential()
        self.droprate = droprate

        for i in range(num_layers):
            layer = DenseBottleNeck(in_channels=self.part2_chnls + i * growth_rate,
                                    growth_rate=growth_rate,
                                    expansion_rate=expansion_rate,
                                    droprate=droprate,
                                    activation=activation)
            self.dense_block.add_module('denselayer_%d' % (i + 1), layer)

        self.spatial_dropout = torch.nn.Dropout2d(p=self.droprate)

    def forward(self, x):

        part1 = x[:, :self.part1_chnls, :, :] #part1 channel 자르기
        part2 = x[:, self.part1_chnls:, :, :] #part2 channel 자르기
        part2 = self.dense_block(part2)
        #part2 = self.spatial_dropout(part2)
        out = torch.cat((part1, part2), 1)

        return out




class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, activation=torch.nn.ReLU):
        super().__init__()
        self.squeeze = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels//reduction_ratio),
            activation(),
            torch.nn.Linear(in_channels//reduction_ratio, in_channels),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x


class SESeparableActivationConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=False, activation=torch.nn.SiLU):
        super(SESeparableActivationConv2d, self).__init__()

        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         groups=in_channels,
                                         bias=bias,
                                         stride=stride,
                                         padding=padding)

        self.pointwise = torch.nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=1,
                                         bias=bias)

        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.activation1 = activation()

        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.activation2 = activation()

        self.seblock = SEBlock(in_channels=out_channels, activation=activation)


    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.activation2(out)
        out = self.seblock(out) * out
        return out


class SEInvertedBottleNect(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_rate=4, stride=1, activation=torch.nn.ReLU6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_rate = expansion_rate
        self.stride = stride

        self.expansion_out = int(self.in_channels * self.expansion_rate)

        self.conv_expansion = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1,
                                                                  in_channels=self.in_channels,
                                                                  out_channels=self.expansion_out,
                                                                  bias=False,
                                                                  stride=1),
                                                  torch.nn.BatchNorm2d(self.expansion_out),
                                                  activation())

        self.conv_depthwise = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=3,
                                                                  in_channels=self.expansion_out,
                                                                  out_channels=self.expansion_out,
                                                                  groups=self.expansion_out,
                                                                  bias=False,
                                                                  padding=1,
                                                                  stride=self.stride),
                                                  torch.nn.BatchNorm2d(self.expansion_out),
                                                  activation())

        self.conv_projection = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1,
                                                                   in_channels=self.expansion_out,
                                                                   out_channels=self.out_channels,
                                                                   bias=False),
                                                   torch.nn.BatchNorm2d(self.out_channels))

        self.seblock = SEBlock(in_channels=self.out_channels, activation=activation)


    def forward(self, x):
        out = self.conv_expansion(x)
        out = self.conv_depthwise(out)
        out = self.conv_projection(out)
        out = self.seblock(out) * out
        if self.stride != 2 and self.in_channels == self.out_channels:
            out = out + x
        return out


class SEDenseBottleNeck(torch.nn.Module):
    def __init__(self, in_channels, growth_rate=32, expansion_rate=4, droprate=0.2, activation=torch.nn.ReLU):
        super().__init__()

        inner_channels = expansion_rate * growth_rate  ##expansion_size=32*4
        self.droprate = droprate

        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False),  ##32*4 #expansion layer
            torch.nn.BatchNorm2d(inner_channels),
            activation(),
            torch.nn.Conv2d(inner_channels, growth_rate, 3, stride=1, padding=1, bias=False),  ##32
            torch.nn.BatchNorm2d(growth_rate),
            activation(),
        )

    def forward(self, x):
        return torch.cat([x, self.residual(x)], 1)


class NCTransition(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sample = torch.nn.Sequential(
            torch.nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        output = self.down_sample(x)
        return output


class ShuffleUnit(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=3,
                 grouped_conv=True,
                 combine='add',
                 activation=torch.nn.ReLU):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.groups = groups
        self.bottleneck_channels = self.in_channels // 4
        self.combine = combine

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self.out_channels -= self.in_channels
            self._combine_func = self._concat
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        if self.grouped_conv == True:
            self.first_1x1_groups = self.groups
        else:
            self.first_1x1_groups = 1

        self.group_compress_convolution = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=self.in_channels,
                            out_channels=self.bottleneck_channels,
                            groups=self.first_1x1_groups,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=self.bottleneck_channels),
            activation()
        )

        self.depthwise_conv = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3,
                            in_channels=self.bottleneck_channels,
                            out_channels=self.bottleneck_channels,
                            stride=self.depthwise_stride,
                            padding=1,
                            groups=self.bottleneck_channels,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=self.bottleneck_channels)
        )

        self.group_expand_convolution = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=self.bottleneck_channels,
                            out_channels=self.out_channels,
                            groups=self.groups,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=self.out_channels)
        )


    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out


    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):

        residual = x

        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)

        output = self.group_compress_convolution(x)
        output = channel_shuffle(output, groups=self.groups)
        output = self.depthwise_conv(output)
        output = self.group_expand_convolution(output)
        output = self._combine_func(residual, output)

        return F.relu(output)


class DarknetResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, activation=torch.nn.LeakyReLU):
        super(DarknetResidualBlock, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            padding=0,
                            bias=False,
                            stride=stride),
            torch.nn.BatchNorm2d(num_features=out_channels),
            activation()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=in_channels),
            activation()
        )

    def forward(self, x):
        skip = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = x + skip
        return x



class HardSwishSEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, activation=torch.nn.ReLU):
        super().__init__()
        self.squeeze = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels//reduction_ratio, bias=False),
            activation(),
            torch.nn.Linear(in_channels//reduction_ratio, in_channels, bias=False),
            torch.nn.Hardswish()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x


class InvertedBottleNectV3(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 expansion_out,
                 out_channels,
                 stride=1,
                 kernel_size=3,
                 use_se=False,
                 activation=torch.nn.ReLU6):
        super().__init__()

        self.in_channels = in_channels
        self.stride = stride
        self.use_se = use_se
        self.expansion_out = expansion_out
        self.out_channels = out_channels
        self.padding = (kernel_size - 1) // 2

        self.conv_expansion = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1,
                                                                  in_channels=self.in_channels,
                                                                  out_channels=self.expansion_out,
                                                                  bias=False,
                                                                  stride=1),
                                                  torch.nn.BatchNorm2d(self.expansion_out),
                                                  activation())

        self.conv_depthwise = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=kernel_size,
                                                                  in_channels=self.expansion_out,
                                                                  out_channels=self.expansion_out,
                                                                  groups=self.expansion_out,
                                                                  bias=False,
                                                                  padding=self.padding,
                                                                  stride=self.stride),
                                                  torch.nn.BatchNorm2d(self.expansion_out),
                                                  activation())

        self.squeeze_layer = HardSwishSEBlock(in_channels=self.expansion_out,
                                              reduction_ratio=4)

        self.conv_projection = torch.nn.Sequential(torch.nn.Conv2d(kernel_size=1,
                                                                   in_channels=self.expansion_out,
                                                                   out_channels=self.out_channels,
                                                                   bias=False),
                                                   torch.nn.BatchNorm2d(self.out_channels))


    def forward(self, x):
        out = self.conv_expansion(x)
        out = self.conv_depthwise(out)

        if self.use_se is True:
            out = self.squeeze_layer(out) * out

        out = self.conv_projection(out)

        if self.stride != 2 and self.in_channels == self.out_channels:
            out = out + x

        return out


class SEConvBlock(torch.nn.Module):
    def __init__(self, in_channels, channels, se_rate=12):
        super(SEConvBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, channels // se_rate, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(channels // se_rate),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels // se_rate, channels, kernel_size=1, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class RexNetLinearBottleNeck(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_se,
                 stride,
                 expand_rate=6,
                 se_rate=12):
        super(RexNetLinearBottleNeck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_channels = self.in_channels * expand_rate
        self.use_se = use_se
        self.stride = stride
        self.se_rate = se_rate

        self.use_skip = self.stride == 1 and self.in_channels <= self.out_channels

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=self.in_channels,
                            out_channels=self.expand_channels,
                            bias=False,
                            stride=1),
            torch.nn.BatchNorm2d(num_features=self.expand_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(kernel_size=3,
                            in_channels=self.expand_channels,
                            out_channels=self.expand_channels,
                            bias=False,
                            stride=self.stride,
                            groups=self.expand_channels,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=self.expand_channels)
        )

        if self.use_se == True:
            self.features.add_module('se_module',
                                     SEConvBlock(in_channels=self.expand_channels,
                                                 channels=self.expand_channels,
                                                 se_rate=self.se_rate))
        self.features.add_module('relu6_layer', torch.nn.ReLU6())
        self.features.add_module('project_conv', torch.nn.Conv2d(kernel_size=1,
                                                                 in_channels=self.expand_channels,
                                                                 out_channels=self.out_channels,
                                                                 bias=False,
                                                                 stride=1))
        self.features.add_module('project_batch_norm', torch.nn.BatchNorm2d(num_features=self.out_channels))

    def forward(self, x):
        out = self.features(x)
        if self.use_skip == True:
            out[:, 0:self.in_channels] += x
        return out




#NFNet Normalization Free Module
_nonlin_gamma = dict(
    identity=1.0,
    celu=1.270926833152771,
    elu=1.2716004848480225,
    gelu=1.7015043497085571,
    leaky_relu=1.70590341091156,
    log_sigmoid=1.9193484783172607,
    log_softmax=1.0002083778381348,
    relu=1.7139588594436646,
    relu6=1.7131484746932983,
    selu=1.0008515119552612,
    sigmoid=4.803835391998291,
    silu=1.7881293296813965,
    softsign=2.338853120803833,
    softplus=1.9203323125839233,
    tanh=1.5939117670059204,
)


_nonlin_table = dict(
    identity=torch.nn.Identity,
    celu=torch.nn.CELU,
    elu=torch.nn.ELU,
    gelu=torch.nn.GELU,
    leaky_relu=torch.nn.LeakyReLU,
    log_sigmoid=torch.nn.LogSigmoid,
    log_softmax=torch.nn.LogSoftmax,
    relu=torch.nn.ReLU,
    relu6=torch.nn.ReLU6,
    selu=torch.nn.SELU,
    sigmoid=torch.nn.Sigmoid,
    silu=torch.nn.SiLU,
    softsign=torch.nn.Softsign,
    softplus=torch.nn.Softplus,
    tanh=torch.nn.Tanh,
)

class GammaActivation(torch.nn.Module):
    def __init__(self,
                 activation='relu',
                 inplace=False):
        super(GammaActivation, self).__init__()

        if activation == 'gelu':
            self.activation = _nonlin_table[activation]()
        else:
            self.activation = _nonlin_table[activation](inplace=inplace)
        self.gamma = _nonlin_gamma[activation]

    def forward(self, x):
        x = self.activation(x) * self.gamma
        return x


class WSConv2d(torch.nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-4,
                 padding_mode='zeros',
                 gain=True,
                 gamma=1.0,
                 use_layernorm=False):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        torch.nn.init.kaiming_normal_(self.weight)
        self.gain = torch.nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        # gamma * 1 / sqrt(fan-in)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps
        # experimental, slightly faster/less GPU memory use
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * \
                F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            mean = torch.mean(
                self.weight, dim=[1, 2, 3], keepdim=True)
            std = torch.std(
                self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, input):
        return F.conv2d(input, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class WSConvTranspose2d(torch.nn.ConvTranspose2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: int = 1,
                 padding_mode: str = 'zeros',
                 gain=True,
                 gamma=1.0,
                 eps=1e-5,
                 use_layernorm=False):

        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         output_padding=output_padding, groups=groups, bias=bias, dilation=dilation,
                         padding_mode=padding_mode)

        torch.nn.init.kaiming_normal_(self.weight)
        self.gain = torch.nn.Parameter(torch.ones(
            self.out_channels, 1, 1, 1)) if gain else None
        # gamma * 1 / sqrt(fan-in)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps
        # experimental, slightly faster/less GPU memory use
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * \
                F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            mean = torch.mean(
                self.weight, dim=[1, 2, 3], keepdim=True)
            std = torch.std(
                self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None, eps: float = 1e-4) -> Tensor:
        return F.conv_transpose2d(input, self.get_weight(), self.bias, self.stride, self.padding, self.output_padding,
                                  self.groups, self.dilation)

class ScaledStdConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, gain=True, gamma=1.0, eps=1e-5, use_layernorm=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gain = torch.nn.Parameter(torch.ones(
            self.out_channels, 1, 1, 1)) if gain else None
        # gamma * 1 / sqrt(fan-in)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps
        # experimental, slightly faster/less GPU memory use
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * \
                F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            mean = torch.mean(
                self.weight, dim=[1, 2, 3], keepdim=True)
            std = torch.std(
                self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class NFSEConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 se_rate=0.5):
        super(NFSEConvBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.hidden_channels = max(1, int(in_channels * se_rate))

        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1),
            torch.nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class NFBasicResidualBlock(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 mid_dim,
                 out_dim,
                 stride=1,
                 groups=32,
                 alpha=0.2,
                 beta=1.0,
                 activation='relu',
                 stochastic_probability=0.25):
        super(NFBasicResidualBlock, self).__init__()

        self.stride = stride
        self.alpha = alpha
        self.beta = beta
        self.groups = groups
        self.channel_per_group = max(1, int(in_dim / self.groups))
        self.features = torch.nn.Sequential(WSConv2d(in_dim,
                                                     mid_dim,
                                                     kernel_size=3,
                                                     stride=self.stride,
                                                     padding=1,
                                                     bias=False,
                                                     groups=self.channel_per_group),
                                            GammaActivation(activation=activation,
                                                            inplace=True),
                                            WSConv2d(mid_dim,
                                                     out_dim,
                                                     kernel_size=3,
                                                     padding='same',
                                                     bias=False),
                                            GammaActivation(activation=activation,
                                                            inplace=True),
                                            NFSEConvBlock(in_channels=out_dim,
                                                          out_channels=out_dim),
                                            StochasticDepth(probability=stochastic_probability))

        self.down_skip_connection = WSConv2d(in_channels=in_dim,
                                             out_channels=out_dim,
                                             kernel_size=1,
                                             stride=self.stride,
                                             bias=False)

    def forward(self, x):
        indentity = x
        if self.stride == 2:
            x = x * self.beta
            down = self.down_skip_connection(indentity)
            out = self.features(x)
            out = out * self.alpha
            out = out + down
            return out
        else:
            x = x * self.beta
            out = self.features(x)
            out = out * self.alpha
            out = out + indentity
            return out


class NFResidualBottleNeck(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 mid_dim,
                 out_dim,
                 stride=1,
                 groups=32,
                 alpha=0.2,
                 beta=1.0,
                 activation='relu',
                 stochastic_probability=0.25):
        super(NFResidualBottleNeck, self).__init__()

        self.stride = stride
        self.alpha = alpha
        self.beta = beta
        self.groups = groups
        self.features = torch.nn.Sequential(WSConv2d(in_dim,
                                                     mid_dim,
                                                     kernel_size=1,
                                                     stride=1,
                                                     bias=True),
                                            GammaActivation(activation=activation,
                                                            inplace=True),
                                            WSConv2d(mid_dim,
                                                     mid_dim,
                                                     kernel_size=3,
                                                     stride=self.stride,
                                                     padding=1,
                                                     bias=True,
                                                     groups=self.groups),
                                            GammaActivation(activation=activation,
                                                            inplace=True),
                                            WSConv2d(mid_dim,
                                                     out_dim,
                                                     kernel_size=1,
                                                     bias=True),
                                            GammaActivation(activation=activation,
                                                            inplace=True),
                                            NFSEConvBlock(in_channels=out_dim,
                                                          out_channels=out_dim),
                                            StochasticDepth(probability=stochastic_probability))

        self.down_skip_connection = WSConv2d(in_channels=in_dim,
                                             out_channels=out_dim,
                                             kernel_size=1,
                                             stride=self.stride,
                                             bias=True)

        self.dim_equalizer = WSConv2d(in_channels=in_dim,
                                      out_channels=out_dim,
                                      kernel_size=1,
                                      stride=self.stride,
                                      bias=True)

    def forward(self, x):
        indentity = x
        if self.stride == 2:
            x = x * self.beta
            down = self.down_skip_connection(indentity)
            out = self.features(x)
            out = out * self.alpha
            out = out + down
            return out

        else:
            x = x * self.beta
            out = self.features(x)
            out = out * self.alpha
            if indentity.size() is not out.size():
                indentity = self.dim_equalizer(indentity)
            out = out + indentity
            return out


class NFSeparableConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=1,
                 stride=1,
                 activation='relu6',
                 bias=False):
        super(NFSeparableConv2d, self).__init__()


        self.depthwise = WSConv2d(in_channels,
                                  in_channels,
                                  kernel_size=kernel_size,
                                  groups=in_channels,
                                  bias=bias,
                                  stride=stride,
                                  padding=padding)

        self.pointwise = WSConv2d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  bias=bias)


        self.activation1 = GammaActivation(activation=activation,
                                           inplace=True)
        self.activation2 = GammaActivation(activation=activation,
                                           inplace=True)


    def forward(self, x):
        out = self.depthwise(x)
        out = self.activation1(out)
        out = self.pointwise(out)
        out = self.activation2(out)
        return out



class NFNetBlock(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 mid_dim,
                 out_dim,
                 stride=1,
                 groups=32,
                 alpha=0.2,
                 beta=1.0,
                 activation='gelu',
                 stochastic_probability=0.25):
        super(NFNetBlock, self).__init__()

        self.stride = stride
        self.alpha = alpha
        self.beta = beta
        self.groups = groups


        self.pre_activation = GammaActivation(activation=activation)

        self.features = torch.nn.Sequential(WSConv2d(in_dim,
                                                     mid_dim,
                                                     kernel_size=1,
                                                     bias=True),
                                            GammaActivation(activation=activation,
                                                            inplace=True),
                                            WSConv2d(mid_dim,
                                                     mid_dim,
                                                     kernel_size=3,
                                                     padding=1,
                                                     stride=self.stride,
                                                     bias=True,
                                                     groups=self.groups),
                                            GammaActivation(activation=activation,
                                                            inplace=True),
                                            WSConv2d(mid_dim,
                                                     mid_dim,
                                                     kernel_size=3,
                                                     padding=1,
                                                     stride=1,
                                                     bias=True,
                                                     groups=self.groups),
                                            GammaActivation(activation=activation,
                                                            inplace=True),
                                            WSConv2d(mid_dim,
                                                     out_dim,
                                                     kernel_size=1,
                                                     bias=True),
                                            NFSEConvBlock(in_channels=out_dim,
                                                          out_channels=out_dim),
                                            StochasticDepth(probability=stochastic_probability))


        self.down_skip_connection = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=3,
                               stride=2,
                               padding=1),
            WSConv2d(in_channels=in_dim,
                     out_channels=out_dim,
                     kernel_size=1,
                     stride=1,
                     bias=True)
        )

    def forward(self, x):
        indentity = x
        if self.stride == 2:
            x = x * self.beta
            x = self.pre_activation(x)
            down = self.down_skip_connection(indentity)
            out = self.features(x)
            out = out * self.alpha
            out = out + down
            return out

        else:
            x = x * self.beta
            x = self.pre_activation(x)
            out = self.features(x)
            out = out * self.alpha
            out = out + indentity
            return out


#NFNet Normalization Free Module



class ResNextResidualBottleNeck(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 inner_channels,
                 out_channels,
                 stride=1,
                 groups=32,
                 activation=torch.nn.ReLU
                 ):
        super().__init__()

        self.stride = stride
        self.channels_per_groups = max(1, int(inner_channels / groups))

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=inner_channels,
                            kernel_size=1,
                            stride=1,
                            bias=False),
            torch.nn.BatchNorm2d(inner_channels),                  ##ex:128
            activation(),
            torch.nn.Conv2d(inner_channels,
                            inner_channels,
                            kernel_size=3,
                            stride=self.stride,
                            padding=1,
                            bias=False,
                            groups=self.channels_per_groups),
            torch.nn.BatchNorm2d(inner_channels),
            activation(),
            torch.nn.Conv2d(inner_channels, out_channels, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),

        )

        self.final_activation = activation()
        self.down_skip_connection = torch.nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=1,
                                                    stride=self.stride)
        self.dim_equalizer = torch.nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1)

    def forward(self, x):
        if self.stride == 2:
            down = self.down_skip_connection(x)
            out = self.features(x)
            out = out + down

        else:
            out = self.features(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        out = self.final_activation(out)
        return out


class StochasticDepth(torch.nn.Module):
    def __init__(self,
                 probability):
        super().__init__()

        self.probability = 1 - probability

    def forward(self, x):
        if self.training:
            pmask = torch.bernoulli(torch.tensor(self.probability))
            x = x * pmask
            #x[:, :, :, :] = pmask
            return x
        else:
            return x







