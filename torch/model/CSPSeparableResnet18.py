import torch
from util.helper import CSPSeparableResidualBlock
from util.helper import SeparableConv2d

class CSPSeparableResnet18(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.SiLU):
        super(CSPSeparableResnet18, self).__init__()

        self.class_num = class_num

        self.conv1 = torch.nn.Sequential(SeparableConv2d(in_channels=3,
                                                         out_channels=64,
                                                         kernel_size=3,
                                                         stride=2,
                                                         padding=1,
                                                         bias=False),
                                         torch.nn.BatchNorm2d(num_features=64),
                                         activation(),
                                         SeparableConv2d(in_channels=64,
                                                         out_channels=64,
                                                         kernel_size=3,
                                                         stride=2,
                                                         padding=1,
                                                         bias=False),
                                         torch.nn.BatchNorm2d(num_features=64),
                                         activation())

        self.conv2 = torch.nn.Sequential(CSPSeparableResidualBlock(in_dim=64, mid_dim=64, out_dim=64, stride=2,
                                                                   activation=activation),
                                         CSPSeparableResidualBlock(in_dim=64, mid_dim=64, out_dim=64, stride=1,
                                                                   activation=activation))

        self.conv3 = torch.nn.Sequential(CSPSeparableResidualBlock(in_dim=64, mid_dim=128, out_dim=128, stride=2,
                                                                   activation=activation),
                                         CSPSeparableResidualBlock(in_dim=128, mid_dim=128, out_dim=128, stride=1,
                                                                   activation=activation))

        self.conv4 = torch.nn.Sequential(CSPSeparableResidualBlock(in_dim=128, mid_dim=256, out_dim=256, stride=2,
                                                                   activation=activation),
                                         CSPSeparableResidualBlock(in_dim=256, mid_dim=256, out_dim=256, stride=1,
                                                                   activation=activation))

        self.conv5 = torch.nn.Sequential(CSPSeparableResidualBlock(in_dim=256, mid_dim=512, out_dim=512, stride=1,
                                                                   activation=activation),
                                         CSPSeparableResidualBlock(in_dim=512, mid_dim=512, out_dim=512, stride=1,
                                                                   activation=activation))

        self.final_conv = SeparableConv2d(in_channels=512,
                                          out_channels=self.class_num,
                                          kernel_size=1,
                                          bias=False,
                                          padding='same')

        self.bn = torch.nn.BatchNorm2d(num_features=self.class_num)
        self.global_average_pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.sigmoid = torch.nn.Sigmoid()

        # module 초기화
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()



    def forward(self, x):

        ##Classificaiton
        conv1 = self.conv1(x)       #256x256x64
        conv2 = self.conv2(conv1)   #128x128x128
        conv3 = self.conv3(conv2)   #64x64x256
        conv4 = self.conv4(conv3)   #32x32x512
        conv5 = self.conv5(conv4)   #16x16x512

        x = self.final_conv(conv5)
        x = self.bn(x)
        x = self.global_average_pooling(x)
        x = x.view([-1, self.class_num])
        x = self.sigmoid(x)
        ##Classificaiton

        return x