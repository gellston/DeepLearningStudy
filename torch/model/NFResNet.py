import torch
from util.helper import WSConv2d
from util.helper import NFBasicResidualBlock


class NFResNet(torch.nn.Module):

    def __init__(self,
                 block_args=[],
                 class_num=5,
                 groups=32,
                 stochastic_probability=0.25):
        super(NFResNet, self).__init__()

        self.class_num = class_num

        self.stem = torch.nn.Sequential(
            WSConv2d(in_channels=3,
                     out_channels=64,
                     stride=2,
                     padding=3,
                     kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3,
                               padding=1,
                               stride=2)
        )

        alpha = 0.2
        expected_std = 1.0
        blocks = []

        for block_index, (in_dim, mid_dim, out_dim, stride) in enumerate(block_args):
            beta = 1. / expected_std
            blocks.append(NFBasicResidualBlock(in_dim=in_dim,
                                               mid_dim=mid_dim,
                                               out_dim=out_dim,
                                               stride=stride,
                                               beta=beta,
                                               alpha=alpha,
                                               groups=groups,
                                               stochastic_probability=stochastic_probability))
            if block_index == 0:
                expected_std = 1.0
            expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

        self.body = torch.nn.Sequential(*blocks)

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Conv2d(in_channels=512,
                                  out_channels=class_num,
                                  kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x


def NFResNet18(class_num=5,
               stochastic_probability=0.25):
    block_args = (
        (64, 64, 64, 1),
        (64, 64, 64, 1),
        (64, 128, 128, 2),
        (128, 128, 128, 1),
        (128, 256, 256, 2),
        (256, 256, 256, 1),
        (256, 512, 512, 2),
        (512, 512, 512, 1),
    )

    return NFResNet(class_num=class_num, stochastic_probability=stochastic_probability, block_args=block_args)