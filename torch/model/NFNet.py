import torch
from util.helper import WSConv2d
from util.helper import NFNetBlock
from util.helper import GammaActivation


class NFNet(torch.nn.Module):

    def __init__(self,
                 block_args=[],
                 class_num=5,
                 gap_dropout_probability=0.25,
                 stochastic_probability=0.25,
                 base_conv=NFNetBlock):
        super(NFNet, self).__init__()

        self.class_num = class_num

        self.stem = torch.nn.Sequential(
            WSConv2d(in_channels=3,
                     out_channels=16,
                     stride=2,
                     padding=1,
                     kernel_size=3,
                     bias=True),
            GammaActivation(activation='gelu',
                            inplace=True),
            WSConv2d(in_channels=16,
                     out_channels=32,
                     stride=1,
                     padding=1,
                     kernel_size=3,
                     bias=True),
            GammaActivation(activation='gelu',
                            inplace=True),
            WSConv2d(in_channels=32,
                     out_channels=64,
                     stride=1,
                     padding=1,
                     kernel_size=3,
                     bias=True),
            GammaActivation(activation='gelu',
                            inplace=True),
            WSConv2d(in_channels=64,
                     out_channels=128,
                     stride=2,
                     padding=1,
                     kernel_size=3,
                     bias=True)
        )

        alpha = 0.2
        expected_std = 1.0
        blocks = []
        num_blocks = len(block_args)
        final_channel = 0
        for block_index, (in_dim, mid_dim, out_dim, stride, cardinality) in enumerate(block_args):
            final_channel = out_dim
            beta = 1. / expected_std
            block_stochastic_probability = stochastic_probability * (block_index + 1) / num_blocks
            blocks.append(base_conv(in_dim=in_dim,
                                    mid_dim=mid_dim,
                                    out_dim=out_dim,
                                    stride=stride,
                                    beta=beta,
                                    alpha=alpha,
                                    groups=cardinality,
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


def NFNetF0(class_num=5,
            gap_dropout_probability=0.25,
            stochastic_probability=0.25):
    block_args = (
        #in, mid, out, stride, cardinality
        (128, 128, 256, 2, 2),
        (256, 256, 512, 2, 2),
        (512, 256, 512, 1, 2),
        (512, 768, 1536, 2, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 2, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
    )
    return NFNet(class_num=class_num,
                 gap_dropout_probability=gap_dropout_probability,
                 stochastic_probability=stochastic_probability,
                 block_args=block_args)


def NFNetF1(class_num=5,
            gap_dropout_probability=0.25,
            stochastic_probability=0.25):
    block_args = (
        #in, mid, out, stride, cardinality
        (128, 128, 256, 2, 1),
        (256, 128, 256, 1, 1),
        (256, 256, 512, 2, 2),
        (512, 256, 512, 1, 2),
        (512, 256, 512, 1, 2),
        (512, 256, 512, 1, 2),
        (512, 768, 1536, 2, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 2, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
    )
    return NFNet(class_num=class_num,
                 gap_dropout_probability=gap_dropout_probability,
                 stochastic_probability=stochastic_probability,
                 block_args=block_args)

def NFNetF2(class_num=5,
            gap_dropout_probability=0.25,
            stochastic_probability=0.25):
    block_args = (
        #in, mid, out, stride, cardinality
        (128, 128, 256, 2, 1),
        (256, 128, 256, 1, 1),
        (256, 128, 256, 1, 1),
        (256, 256, 512, 2, 2),
        (512, 256, 512, 1, 2),
        (512, 256, 512, 1, 2),
        (512, 256, 512, 1, 2),
        (512, 256, 512, 1, 2),
        (512, 256, 512, 1, 2),
        (512, 768, 1536, 2, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 1, 6),
        (1536, 768, 1536, 2, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
        (1536, 768, 1536, 1, 3),
    )
    return NFNet(class_num=class_num,
                 gap_dropout_probability=gap_dropout_probability,
                 stochastic_probability=stochastic_probability,
                 block_args=block_args)



