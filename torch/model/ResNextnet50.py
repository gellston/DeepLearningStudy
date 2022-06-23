import torch
from util.helper import ResNextResidualBottleNeck

class ResNextnet50(torch.nn.Module):

    def __init__(self,
                 class_num=5,
                 groups=32,
                 activation=torch.nn.SiLU):
        super(ResNextnet50, self).__init__()

        self.class_num = class_num

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=64,
                            stride=2,
                            padding=3,
                            kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3,
                               padding=1,
                               stride=2)
        )

        block_args = (
            (64, 64, 256, 1),
            (256, 64, 256, 1),
            (256, 64, 256, 2),

            (256, 128, 512, 1),
            (512, 128, 512, 1),
            (512, 128, 512, 1),
            (512, 128, 512, 2),

            (512, 256, 1024, 1),
            (1024, 256, 1024, 1),
            (1024, 256, 1024, 1),
            (1024, 256, 1024, 1),
            (1024, 256, 1024, 1),
            (1024, 256, 1024, 2),

            (1024, 512, 2048, 1),
            (2048, 512, 2048, 1),
            (2048, 512, 2048, 1),
        )

        blocks = []
        for block_index, (in_dim, mid_dim, out_dim, stride) in enumerate(block_args):
            blocks.append(ResNextResidualBottleNeck(
                in_channels=in_dim,
                inner_channels=mid_dim,
                out_channels=out_dim,
                stride=stride,
                groups=groups,
                activation=activation
            ))
        self.body = torch.nn.Sequential(*blocks)
        self.global_average_pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.final_conv = torch.nn.Conv2d(kernel_size=1,
                                          in_channels=2048,
                                          out_channels=class_num)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.global_average_pooling(x)
        x = self.final_conv(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)

        return x