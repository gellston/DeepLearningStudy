import torch
import torch.nn.functional as F
import torch.nn as nn
from util.helper import KShopResnet
from util.helper import KShopSEBlock

class KShopNet(torch.nn.Module):

    def __init__(self,
                 activation=torch.nn.SiLU,
                 inner_channel=32,
                 exapnd_rate=1,
                 se_rate=0.5,
                 layer_length=2):
        super(KShopNet,
              self).__init__()

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=57,
                            out_channels=inner_channel,
                            bias=False),
            torch.nn.BatchNorm2d(inner_channel),

        )



        block = []
        for i in range(layer_length):
            block.append(KShopResnet(in_channels=inner_channel,
                                       out_channels=inner_channel,
                                       expand_rate=exapnd_rate,
                                       activation=activation))
        self.feature1 = torch.nn.Sequential(*block)


        block = []
        for i in range(layer_length):
            block.append(KShopResnet(in_channels=inner_channel,
                                       out_channels=inner_channel,
                                       expand_rate=exapnd_rate,
                                       activation=activation))
        self.feature2 = torch.nn.Sequential(*block)



        block = []
        for i in range(layer_length):
            block.append(KShopResnet(in_channels=inner_channel,
                                       out_channels=inner_channel,
                                       expand_rate=exapnd_rate,
                                       activation=activation))
        self.feature3 = torch.nn.Sequential(*block)


        block = []
        for i in range(layer_length):
            block.append(KShopResnet(in_channels=inner_channel,
                                       out_channels=inner_channel,
                                       expand_rate=exapnd_rate,
                                       activation=activation))
        self.feature4 = torch.nn.Sequential(*block)



        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=(4, 1),
                            in_channels=inner_channel,
                            out_channels=1,
                            bias=False),
            torch.nn.ReLU()
        )

        # module 초기화
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()


    def forward(self, x):
        x = self.stem(x)
        x1 = self.feature1(x)
        x2 = self.feature2(x)
        x3 = self.feature3(x)
        x4 = self.feature4(x)
        x = torch.cat([x1, x2, x3, x4], dim=2)
        x = self.final(x)
        x = x.view([-1, 1])

        return x