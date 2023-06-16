import torch

from util.helper import SelfAttResBlock


class NonLocalNet18(torch.nn.Module):

    def __init__(self,
                 block_args=[],
                 class_num=5,
                 activation=torch.nn.ReLU,
                 gap_dropout_probability=0.25):
        super(NonLocalNet18, self).__init__()

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

        self.layer1 = torch.nn.Sequential(
            SelfAttResBlock(in_channel=64,
                            out_channel=64,
                            stride=1,
                            activation=activation),
            SelfAttResBlock(in_channel=64,
                            out_channel=64,
                            stride=2,
                            activation=activation),
        )

        self.layer2 = torch.nn.Sequential(
            SelfAttResBlock(in_channel=64,
                            out_channel=128,
                            stride=1,
                            activation=activation),
            SelfAttResBlock(in_channel=128,
                            out_channel=128,
                            stride=2,
                            activation=activation),
        )

        self.layer3 = torch.nn.Sequential(
            SelfAttResBlock(in_channel=128,
                            out_channel=256,
                            stride=1,
                            activation=activation),
            SelfAttResBlock(in_channel=256,
                            out_channel=256,
                            stride=2,
                            activation=activation),
        )

        self.layer4 = torch.nn.Sequential(
            SelfAttResBlock(in_channel=256,
                            out_channel=512,
                            stride=1,
                            activation=activation),
            SelfAttResBlock(in_channel=512,
                            out_channel=512,
                            stride=2,
                            activation=activation),
        )


        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc_dropout = torch.nn.Dropout2d(p=gap_dropout_probability)
        self.fc = torch.nn.Conv2d(in_channels=512,
                                  out_channels=class_num,
                                  kernel_size=1)
        torch.nn.init.normal_(self.fc.weight, mean=0, std=0.01)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = self.fc_dropout(x)
        x = self.fc(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x
