import torch
from util.helper import ResidualBlock

class Resnet18(torch.nn.Module):

    def __init__(self, class_num=5):
        super(Resnet18, self).__init__()

        self.class_num = class_num

        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3,
                                                         out_channels=64,
                                                         kernel_size=7,
                                                         stride=2),
                                         torch.nn.BatchNorm2d(num_features=64),
                                         torch.nn.SiLU(),
                                         torch.nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv2 = torch.nn.Sequential(ResidualBlock(in_dim=64, mid_dim=64, out_dim=64, stride=1),
                                         ResidualBlock(in_dim=64, mid_dim=64, out_dim=64, stride=1))

        self.conv3 = torch.nn.Sequential(ResidualBlock(in_dim=64, mid_dim=128, out_dim=128, stride=2),
                                         ResidualBlock(in_dim=128, mid_dim=128, out_dim=128, stride=1))

        self.conv4 = torch.nn.Sequential(ResidualBlock(in_dim=128, mid_dim=256, out_dim=256, stride=2),
                                         ResidualBlock(in_dim=256, mid_dim=256, out_dim=256, stride=1))

        self.conv5 = torch.nn.Sequential(ResidualBlock(in_dim=256, mid_dim=512, out_dim=512, stride=2),
                                         ResidualBlock(in_dim=512, mid_dim=512, out_dim=512, stride=1))

        self.final_conv = torch.nn.Conv2d(in_channels=512,
                                          out_channels=self.class_num,
                                          kernel_size=3,
                                          padding='same')

        self.bn = torch.nn.BatchNorm2d(num_features=self.class_num)

        self.global_average_pooling = torch.nn.AdaptiveAvgPool2d(1)

        self.sigmoid = torch.nn.Sigmoid()



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.final_conv(x)
        x = self.bn(x)
        x = self.global_average_pooling(x)
        x = x.view([-1, self.class_num])
        x = self.sigmoid(x)

        return x