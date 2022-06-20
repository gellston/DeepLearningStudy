import torch
import torch.nn.functional as F
from util.helper import NFSeparableConv2d
from util.helper import WSConv2d


class NFMobileNetV1(torch.nn.Module):

    def __init__(self, class_num=5):
        super(NFMobileNetV1, self).__init__()

        self.class_num = class_num

        block_args = (
            (32, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 512, 2),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 1024, 2),
            (1024, 1024, 1),
        )

        self.stem = torch.nn.Sequential(
            WSConv2d(in_channels=3,
                     out_channels=32,
                     stride=2,
                     padding=1,
                     kernel_size=3),
            torch.nn.ReLU6()
        )

        blocks = []
        for block_index, (in_dim, out_dim, stride) in enumerate(block_args):
            blocks.append(NFSeparableConv2d(in_channels=in_dim,
                                            out_channels=out_dim,
                                            stride=stride,
                                            kernel_size=3,
                                            bias=False))

        self.body = torch.nn.Sequential(*blocks)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Conv2d(kernel_size=1,
                                  in_channels=1024,
                                  out_channels=self.class_num,
                                  bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x