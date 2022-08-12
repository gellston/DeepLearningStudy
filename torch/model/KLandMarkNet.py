import torch
from util.helper import WSConv2d
from util.helper import NFKLandMarkResidualBlock
from util.helper import GammaActivation


class KLandMarkNet(torch.nn.Module):

    def __init__(self,
                 block_args=[],
                 class_num=5,
                 se_rate=0.07,
                 gap_dropout_probability=0.25,
                 stochastic_probability=0.25,
                 block_dropout_probability=0.25,
                 base_conv=NFKLandMarkResidualBlock):
        super(KLandMarkNet, self).__init__()

        self.class_num = class_num

        self.stem = torch.nn.Sequential(
            WSConv2d(in_channels=3,
                     out_channels=32,
                     padding=2,
                     stride=2,
                     kernel_size=7,
                     bias=True),
            GammaActivation(activation='silu',
                            inplace=True)
        )

        alpha = 0.2
        expected_std = 1.0
        blocks = []
        num_blocks = len(block_args)
        final_channel = 0
        for block_index, (in_dim, mid_dim, out_dim, stride, kernel_size, dilation_rate) in enumerate(block_args):
            final_channel = out_dim
            beta = 1. / expected_std
            block_stochastic_probability = stochastic_probability * (block_index + 1) / num_blocks
            blocks.append(base_conv(in_dim=in_dim,
                                    mid_dim=mid_dim,
                                    out_dim=out_dim,
                                    stride=stride,
                                    beta=beta,
                                    alpha=alpha,
                                    se_rate=se_rate,
                                    activation='silu',
                                    kernel_size=kernel_size,
                                    dilation_rate=dilation_rate,
                                    block_dropout_probability=block_dropout_probability,
                                    stochastic_probability=block_stochastic_probability))
            if block_index == 0:
                expected_std = 1.0
            expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

        self.body = torch.nn.Sequential(*blocks)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc_dropout = torch.nn.Dropout2d(p=gap_dropout_probability)
        self.fc = torch.nn.Conv2d(in_channels=final_channel,
                                  out_channels=class_num,
                                  kernel_size=1)
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


def KLandMarkNet18(class_num=5,
                   gap_dropout_probability=0.25,
                   block_dropout_probability=0.25,
                   stochastic_probability=0.25):
    block_args = (
        (32, 32, 32, 1, 7, 1), #240x160
        (32, 32, 32, 1, 7, 1),
        (32, 32, 32, 1, 7, 1),
        (32, 64, 64, 2, 5, 1), #120x80
        (64, 64, 64, 1, 5, 1),
        (64, 64, 64, 1, 5, 1),
        (64, 128, 128, 2, 5, 2), #60x40
        (128, 128, 128, 1, 5, 2),
        (128, 128, 128, 1, 5, 2),
        (128, 256, 256, 2, 3, 1), #30x20
        (256, 256, 256, 1, 3, 1),
        (256, 256, 256, 1, 3, 1),
        (256, 256, 512, 2, 3, 1), #15x10
        (512, 512, 512, 1, 3, 1),
        (512, 512, 512, 1, 3, 1),
        (512, 512, 512, 2, 3, 1), #7x5
        (512, 512, 512, 1, 3, 1),
        (512, 512, 512, 1, 3, 1),
    )
    return KLandMarkNet(class_num=class_num,
                        block_dropout_probability=block_dropout_probability,
                        gap_dropout_probability=gap_dropout_probability,
                        stochastic_probability=stochastic_probability,
                        block_args=block_args)


