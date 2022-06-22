import torch
import torch.nn.functional as F
from util.helper import ShuffleUnit


class ShuffleNet(torch.nn.Module):

    def __init__(self, groups=3, class_num=2, activation=torch.nn.ReLU):
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.class_num = class_num

        self.preconv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=24,
                            kernel_size=3,
                            stride=2,
                            bias=True,
                            padding=1),
            torch.nn.MaxPool2d(kernel_size=3,
                               stride=2,
                               padding=1)
        )

        self.stage2 = torch.nn.Sequential(
            ShuffleUnit(in_channels=24,
                        out_channels=240,
                        grouped_conv=True, ##Just for food dataset originally it should be false
                        combine='concat',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=240,
                        out_channels=240,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=240,
                        out_channels=240,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=240,
                        out_channels=240,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation)
        )

        self.stage3 = torch.nn.Sequential(
            ShuffleUnit(in_channels=240,
                        out_channels=480,
                        grouped_conv=True,
                        combine='concat',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=480,
                        out_channels=480,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=480,
                        out_channels=480,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=480,
                        out_channels=480,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=480,
                        out_channels=480,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=480,
                        out_channels=480,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=480,
                        out_channels=480,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=480,
                        out_channels=480,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation)
        )

        self.stage4 = torch.nn.Sequential(
            ShuffleUnit(in_channels=480,
                        out_channels=960,
                        grouped_conv=True,
                        combine='concat',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=960,
                        out_channels=960,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation),
            ShuffleUnit(in_channels=960,
                        out_channels=960,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups),
            ShuffleUnit(in_channels=960,
                        out_channels=960,
                        grouped_conv=True,
                        combine='add',
                        groups=self.groups,
                        activation=activation)
        )


        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear = torch.nn.Linear(in_features=960,
                                      out_features=self.class_num,
                                      bias=True)


        # module 초기화
        self.initialize_weights()

    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.preconv(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_avg_pool(x)
        x = x.view([-1, 960])
        x = self.linear(x)
        x = torch.softmax(x, dim=1)
        return x