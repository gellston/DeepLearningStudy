import torch
from util.helper import ResidualBottleNeck
from util.helper import ResidualBlock


class ResNet(torch.nn.Module):

    def __init__(self,
                 block_args=[],
                 class_num=5,
                 activation=torch.nn.ReLU,
                 gap_dropout_probability=0.25,
                 base_conv=ResidualBlock):
        super(ResNet, self).__init__()

        self.class_num = class_num

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=64,
                            stride=2,
                            padding=3,
                            kernel_size=7,
                            bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3,
                               padding=1,
                               stride=2)
        )

        blocks = []
        final_channel = 0
        for block_index, (in_dim, mid_dim, out_dim, stride) in enumerate(block_args):
            blocks.append(base_conv(in_channels=in_dim,
                                    inner_channels=mid_dim,
                                    out_channels=out_dim,
                                    stride=stride,
                                    activation=activation))
            final_channel = out_dim

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


def ResNet18(class_num=5,
             gap_dropout_probability=0.25,
             activation=torch.nn.ReLU):
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
    return ResNet(class_num=class_num,
                  activation=activation,
                  gap_dropout_probability=gap_dropout_probability,
                  block_args=block_args)


def ResNet34(class_num=5,
             gap_dropout_probability=0.25,
             activation=torch.nn.ReLU):
    block_args = (
        (64, 64, 64, 1),
        (64, 64, 64, 1),
        (64, 64, 64, 1),
        (64, 128, 128, 2),
        (128, 128, 128, 1),
        (128, 128, 128, 1),
        (128, 256, 256, 2),
        (256, 256, 256, 1),
        (256, 256, 256, 1),
        (256, 256, 256, 1),
        (256, 256, 256, 1),
        (256, 256, 256, 1),
        (256, 512, 512, 2),
        (512, 512, 512, 1),
        (512, 512, 512, 1),
    )
    return ResNet(class_num=class_num,
                  activation=activation,
                  gap_dropout_probability=gap_dropout_probability,
                  block_args=block_args)


def ResNet50(class_num=5,
             gap_dropout_probability=0.25,
             activation=torch.nn.ReLU):
    block_args = (
        (64, 64, 256, 2),
        (256, 64, 256, 1),
        (256, 64, 256, 1),
        (256, 128, 512, 2),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 256, 1024, 2),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 512, 2048, 2),
        (2048, 512, 2048, 1),
        (2048, 512, 2048, 1),
    )
    return ResNet(class_num=class_num,
                  activation=activation,
                  gap_dropout_probability=gap_dropout_probability,
                  block_args=block_args,
                  base_conv=ResidualBottleNeck)


def ResNet101(class_num=5,
              gap_dropout_probability=0.25,
              activation=torch.nn.ReLU):
    block_args = (
        (64, 64, 256, 2),
        (256, 64, 256, 1),
        (256, 64, 256, 1),
        (256, 128, 512, 2),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 256, 1024, 2),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 512, 2048, 2),
        (2048, 512, 2048, 1),
        (2048, 512, 2048, 1),
    )
    return ResNet(class_num=class_num,
                  activation=activation,
                  gap_dropout_probability=gap_dropout_probability,
                  block_args=block_args,
                  base_conv=ResidualBottleNeck)


def ResNet152(class_num=5,
              gap_dropout_probability=0.25,
              activation=torch.nn.ReLU):
    block_args = (
        (64, 64, 256, 2),
        (256, 64, 256, 1),
        (256, 64, 256, 1),
        (256, 128, 512, 2),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 128, 512, 1),
        (512, 256, 1024, 2),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 256, 1024, 1),
        (1024, 512, 2048, 2),
        (2048, 512, 2048, 1),
        (2048, 512, 2048, 1),
    )
    return ResNet(class_num=class_num,
                  activation=activation,
                  gap_dropout_probability=gap_dropout_probability,
                  block_args=block_args,
                  base_conv=ResidualBottleNeck)