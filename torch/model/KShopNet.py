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
                 layer_length=2,
                 unit_channel_rate=1.2):
        super(KShopNet,
              self).__init__()

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=57,
                            out_channels=inner_channel,
                            bias=False),
            torch.nn.BatchNorm2d(inner_channel),
            activation(),
            KShopSEBlock(in_channels=inner_channel,
                         out_channels=inner_channel,
                         se_rate=se_rate)
        )

        blocks = []

        unit_channel = int(inner_channel / layer_length)
        temp_channel = inner_channel

        for i in range(layer_length):
            blocks.append(KShopResnet(in_channels=temp_channel,
                                      out_channels=temp_channel,
                                      expand_rate=exapnd_rate,
                                      se_rate=se_rate,
                                      activation=activation))

            final_temp_channel = temp_channel - int(unit_channel / unit_channel_rate)

            blocks.append(torch.nn.Conv2d(in_channels=temp_channel,
                                          out_channels=final_temp_channel,
                                          kernel_size=1,
                                          bias=False))
            blocks.append(torch.nn.BatchNorm2d(final_temp_channel))
            blocks.append(activation())
            blocks.append(KShopSEBlock(in_channels=final_temp_channel,
                                       out_channels=final_temp_channel,
                                       se_rate=se_rate))

            temp_channel = final_temp_channel

            print('decreasing channel number =', temp_channel)

        self.feature = torch.nn.Sequential(*blocks)


        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=temp_channel,
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
        x = self.feature(x)
        x = self.final(x)
        x = x.view([-1, 1])

        return x