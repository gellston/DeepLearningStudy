import torch
import torch.nn.functional as F
import torch.nn as nn
from util.helper import KShopResnet


class KShopNetV3(torch.nn.Module):

    def __init__(self,
                 activation=torch.nn.ReLU,
                 expand_rate=0.5,
                 dropout_rate=0.3):
        super(KShopNetV3,
              self).__init__()



        self.stem = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=57),
            torch.nn.Conv2d(in_channels=57,
                            out_channels=16,
                            kernel_size=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=16),
            activation()
        )

        self.features = torch.nn.Sequential(
            KShopResnet(in_channels=16,
                        out_channels=8,
                        expand_rate=expand_rate,
                        activation=activation,
                        dropout_rate=dropout_rate,
                        stochastic_probability=0.125),
            KShopResnet(in_channels=8,
                        out_channels=4,
                        expand_rate=expand_rate,
                        activation=activation,
                        dropout_rate=dropout_rate,
                        stochastic_probability=0.25),
        )

        self.final_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4,
                            out_channels=1,
                            bias=True,
                            kernel_size=1),
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
        x = self.features(x)
        x = self.final_conv(x)
        x = x.view([-1, 1])
        return x