import torch
import torch.nn.functional as F
from util.helper import SeparableActivationConv2d
from util.helper import InvertedBottleNectV3
from util.helper import CBAM


class MWNetV7_224_1(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.ReLU6):
        super(MWNetV7_224_1, self).__init__()

        self.class_num = class_num

        self.features = torch.nn.Sequential(
            # 224
            torch.nn.Conv2d(in_channels=1,
                            out_channels=32,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation(),
            #112
            InvertedBottleNectV3(in_channels=32,
                                 expansion_out=64,
                                 out_channels=32,
                                 stride=2,
                                 use_se=True,  # 200x200 3
                                 kernel_size=3,
                                 activation=activation),
            # 64
            InvertedBottleNectV3(in_channels=32,
                                 expansion_out=64,
                                 out_channels=32,
                                 stride=2,
                                 use_se=True,  # 200x200 3
                                 kernel_size=3,
                                 activation=activation),

            # 32
            InvertedBottleNectV3(in_channels=32,
                                 expansion_out=64,
                                 out_channels=64,
                                 stride=2,
                                 use_se=True,  # 200x200 3
                                 kernel_size=3,
                                 activation=activation),
            # 16
            InvertedBottleNectV3(in_channels=64,
                                 expansion_out=128,
                                 out_channels=64,
                                 stride=2,
                                 use_se=True,  # 200x200 3
                                 kernel_size=3,
                                 activation=activation),
            # 8
            InvertedBottleNectV3(in_channels=64,
                                 expansion_out=128,
                                 out_channels=64,
                                 stride=2,
                                 use_se=True,  # 200x200 3
                                 kernel_size=3,
                                 activation=activation),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels=64,
                            out_channels=self.class_num,
                            kernel_size=1,
                            bias=True),


        )

        # module 초기화
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x