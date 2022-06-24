import torch
import torch.nn.functional as F

from util.helper import InvertedBottleNeck




class MobileVitBlock(torch.nn.Module):
    def __init__(self,
                 dim=64,
                 depth=2,
                 channel=32,
                 kernel_size=3,
                 patch_size=(2, 2),
                 mlp_dim=64*2):

        self.conv1 = torch.nn.BatchNorm2d(
            torch.nn.Conv2d(kernel_size=kernel_size,
                            in_channels=channel,
                            out_channels=channel,
                            bias=False,
                            padding=0)
        )

    def forward(self, x):

        return x





class MobileVit(torch.nn.Module):

    def __init__(self,
                 class_num=5,
                 expansion_rate=2,
                 activation=torch.nn.SiLU):

        super(MobileVit, self).__init__()

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.SiLU()
        )

        dims = [64, 80, 96]
        L = [2, 4, 3]

        self.conv_features = torch.nn.Sequential(

            InvertedBottleNeck(in_channels=16,
                               out_channels=16,
                               expansion_rate=expansion_rate,
                               stride=1,
                               activation=activation),

            InvertedBottleNeck(in_channels=16,
                               out_channels=24,
                               expansion_rate=expansion_rate,
                               stride=2,
                               activation=activation),

            InvertedBottleNeck(in_channels=24,
                               out_channels=24,
                               expansion_rate=expansion_rate,
                               stride=1,
                               activation=activation),

            InvertedBottleNeck(in_channels=24,
                               out_channels=48,
                               expansion_rate=expansion_rate,
                               stride=1,
                               activation=activation),

            InvertedBottleNeck(in_channels=48,
                               out_channels=48,
                               expansion_rate=expansion_rate,
                               stride=2,
                               activation=activation),

            InvertedBottleNeck(in_channels=48,
                               out_channels=64,
                               expansion_rate=expansion_rate,
                               stride=2,
                               activation=activation),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv_features(x)

        print('shape x = ', x.shape)

        return x