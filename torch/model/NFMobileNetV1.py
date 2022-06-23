import torch

from util.helper import NFSeparableConv2d
from util.helper import WSConv2d
from util.helper import GammaActivation

class NFMobileNetV1(torch.nn.Module):

    def __init__(self,
                 class_num=5,
                 gap_dropout_probability=0.25,
                 dropblock_probability=0.2):
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
                     kernel_size=3,
                     bias=True),
            GammaActivation(activation='relu6')
        )

        blocks = []
        num_blocks = len(block_args)
        for block_index, (in_dim, out_dim, stride) in enumerate(block_args):
            block_dropblock_probability = dropblock_probability * (block_index + 1) / num_blocks
            blocks.append(NFSeparableConv2d(in_channels=in_dim,
                                            out_channels=out_dim,
                                            stride=stride,
                                            kernel_size=3,
                                            activation='relu6',
                                            bias=True))
            if block_index > 8:
                blocks.append(torch.nn.Dropout2d(p=block_dropblock_probability)) #Dropblock 적용

        self.body = torch.nn.Sequential(*blocks)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc_dropout = torch.nn.Dropout2d(p=gap_dropout_probability)
        self.fc = torch.nn.Conv2d(kernel_size=1,
                                  in_channels=1024,
                                  out_channels=self.class_num,
                                  bias=False)
        torch.nn.init.normal_(self.fc.weight, mean=0, std=0.01)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.gap(x)
        x = self.fc_dropout(x)
        x = self.fc(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x