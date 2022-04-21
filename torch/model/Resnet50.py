import torch
from util.helper import ResidualBottleNeck

class Resnet50(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.SiLU):
        super(Resnet50, self).__init__()

        self.class_num = class_num

        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3,
                                                         out_channels=64,
                                                         kernel_size=3,
                                                         stride=2,
                                                         padding=1,
                                                         bias=False),
                                         torch.nn.BatchNorm2d(num_features=64),
                                         activation(),
                                         torch.nn.Conv2d(in_channels=64,
                                                         out_channels=64,
                                                         kernel_size=3,
                                                         stride=2,
                                                         padding=1,
                                                         bias=False),
                                         torch.nn.BatchNorm2d(num_features=64),
                                         activation())

        self.conv2 = torch.nn.Sequential(ResidualBottleNeck(in_channels=64,
                                                            inner_channels=64,
                                                            out_channels=256,
                                                            stride=2,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=256,
                                                            inner_channels=64,
                                                            out_channels=256,
                                                            stride=1,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=256,
                                                            inner_channels=64,
                                                            out_channels=256,
                                                            stride=1,
                                                            activation=activation))

        self.conv3 = torch.nn.Sequential(ResidualBottleNeck(in_channels=256,
                                                            inner_channels=128,
                                                            out_channels=512,
                                                            stride=2,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=512,
                                                            inner_channels=128,
                                                            out_channels=512,
                                                            stride=1,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=512,
                                                            inner_channels=128,
                                                            out_channels=512,
                                                            stride=1,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=512,
                                                            inner_channels=128,
                                                            out_channels=512,
                                                            stride=1,
                                                            activation=activation))

        self.conv4 = torch.nn.Sequential(ResidualBottleNeck(in_channels=512,
                                                            inner_channels=256,
                                                            out_channels=1024,
                                                            stride=2,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=1024,
                                                            inner_channels=256,
                                                            out_channels=1024,
                                                            stride=1,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=1024,
                                                            inner_channels=256,
                                                            out_channels=1024,
                                                            stride=1,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=1024,
                                                            inner_channels=256,
                                                            out_channels=1024,
                                                            stride=1,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=1024,
                                                            inner_channels=256,
                                                            out_channels=1024,
                                                            stride=1,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=1024,
                                                            inner_channels=256,
                                                            out_channels=1024,
                                                            stride=1,
                                                            activation=activation))

        self.conv5 = torch.nn.Sequential(ResidualBottleNeck(in_channels=1024,
                                                            inner_channels=512,
                                                            out_channels=2048,
                                                            stride=1,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=2048,
                                                            inner_channels=512,
                                                            out_channels=2048,
                                                            stride=1,
                                                            activation=activation),
                                         ResidualBottleNeck(in_channels=2048,
                                                            inner_channels=512,
                                                            out_channels=2048,
                                                            stride=1,
                                                            activation=activation))
        self.final_conv = torch.nn.Conv2d(in_channels=2048,
                                          out_channels=self.class_num,
                                          kernel_size=1,
                                          bias=True)
        self.final_bn = torch.nn.BatchNorm2d(num_features=self.class_num)
        self.global_average_pooling = torch.nn.AdaptiveAvgPool2d(1)



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.global_average_pooling(x)
        x = x.view([-1, self.class_num])
        x = torch.sigmoid(x)

        return x