import torch

from util.helper import WideResidualNetworkBlock


class WideResNet(torch.nn.Module):

    def __init__(self,
                 depth=16,
                 class_num=5,
                 widen_factor=8,
                 activation=torch.nn.ReLU,
                 dropout_probability=0.25):
        super(WideResNet, self).__init__()

        self.class_num = class_num

        nChannels = [16, 16*widen_factor, 32* widen_factor, 64*widen_factor]

        assert((depth - 4) % 6 == 0)

        n = (depth - 4) / 6

        self.stem = torch.nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = WideResidualNetworkBlock(n, in_channels=nChannels[0], out_channels=nChannels[1], stride=1, dropRate=dropout_probability, activation=activation)
        self.layer2 = WideResidualNetworkBlock(n, in_channels=nChannels[1], out_channels=nChannels[2], stride=2, dropRate=dropout_probability, activation=activation)
        self.layer3 = WideResidualNetworkBlock(n, in_channels=nChannels[2], out_channels=nChannels[3], stride=2, dropRate=dropout_probability, activation=activation)
  
        self.bn1 = torch.nn.BatchNorm2d(nChannels[3])
        self.activation = activation()
        self.gap = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Conv2d(in_channels=nChannels[3], out_channels=class_num, kernel_size=1)
        

        

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x

