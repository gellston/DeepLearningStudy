import torch
import numpy as np
import torch.nn.functional as F

def IOU(target, prediction):
    prediction = np.where(prediction > 0.5, 1, 0)
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


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

        self.stride = stride;
        self.residual_block = torch.nn.Sequential(torch.nn.Conv2d(in_dim,
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

        self.projection = torch.nn.Conv2d(in_channels=in_dim,
                                          out_channels=out_dim,
                                          kernel_size=1,
                                          stride=2)
        self.activation = activation()



    def forward(self, x):
        out = self.residual_block(x)  # F(x)
        if self.stride == 2:
            out = torch.add(out, self.projection(x))
        else:
            out = torch.add(out, x)
        out = self.activation(out)
        return out


class BottleNeck(torch.nn.Module):
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
        #output = self.spatial_dropout(output)
        #if self.droprate > 0:
        #    output = F.dropout(output, p=self.droprate, inplace=False, training=self.training)
        return output


class DenseBlock(torch.nn.Module):

    def __init__(self, num_input_features, num_layers, expansion_rate, growth_rate, droprate, activation=torch.nn.ReLU):
        super(DenseBlock, self).__init__()

        self.dense_block = torch.nn.Sequential()
        self.droprate = droprate

        for i in range(num_layers):
            layer = BottleNeck(in_channels=num_input_features + i * growth_rate,
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

        self.stride = stride;
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
                                           stride=self.stride)

        self.projection2 = torch.nn.Conv2d(in_channels=self.part1_chnls,
                                          out_channels=self.part1_out_chnls,
                                          kernel_size=1,
                                          stride=self.stride)

        self.activation = activation()

    def forward(self, x):

        part1 = x[:, :self.part1_chnls, :, :] #part1 channel 자르기
        part2 = x[:, self.part1_chnls:, :, :] #part2 channel 자르기
        skip_connection = part2

        if self.stride == 2:
            skip_connection = self.projection1(skip_connection)
            part1 = self.projection2(part1)

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
            layer = BottleNeck(in_channels=self.part2_chnls + i * growth_rate,
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



class CSPSeparableResidualBlock(torch.nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim, stride=1, part_ratio=0.5, activation=torch.nn.SiLU):
        super(CSPSeparableResidualBlock, self).__init__()

        self.check_different = mid_dim != out_dim

        self.part1_chnls = int(in_dim * part_ratio)
        self.part2_chnls = in_dim - self.part1_chnls                ##Residual Layer Channel Calculation

        self.part1_out_chnls = int(out_dim * part_ratio)
        self.part2_out_chnls = out_dim - self.part1_out_chnls

        self.stride = stride;
        self.residual_block = torch.nn.Sequential(SeparableConv2d(self.part2_chnls,
                                                                  mid_dim,
                                                                  kernel_size=3,
                                                                  stride=self.stride,
                                                                  padding=1,
                                                                  bias=False),
                                                  torch.nn.BatchNorm2d(num_features=mid_dim),
                                                  activation(),
                                                  SeparableConv2d(mid_dim,
                                                                  self.part2_out_chnls,
                                                                  kernel_size=3,
                                                                  padding='same',
                                                                  bias=False),
                                                  torch.nn.BatchNorm2d(num_features=self.part2_out_chnls))

        self.projection1 = torch.nn.Conv2d(in_channels=self.part2_chnls,            ##Residual Projection
                                           out_channels=self.part2_out_chnls,
                                           kernel_size=1,
                                           stride=self.stride)

        self.projection2 = torch.nn.Conv2d(in_channels=self.part1_chnls,
                                          out_channels=self.part1_out_chnls,
                                          kernel_size=1,
                                          stride=self.stride)

        self.activation = activation()

    def forward(self, x):

        part1 = x[:, :self.part1_chnls, :, :] #part1 channel 자르기
        part2 = x[:, self.part1_chnls:, :, :] #part2 channel 자르기
        skip_connection = part2

        if self.stride == 2 or self.check_different:
            skip_connection = self.projection1(skip_connection)
            part1 = self.projection2(part1)

        residual = self.residual_block(part2)  # F(x)
        residual = torch.add(residual, skip_connection)
        residual = self.activation(residual)

        out = torch.cat((part1, residual), 1)

        return out

