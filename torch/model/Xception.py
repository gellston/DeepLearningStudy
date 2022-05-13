import torch
from util.helper import SeparableActivationConv2d
from util.helper import SeparableConv2d

class MiddleFlow(torch.nn.Module):
    def __init__(self, activation=torch.nn.ReLU):
        super(MiddleFlow, self).__init__()
        self.features = torch.nn.Sequential(
            activation(),
            SeparableConv2d(in_channels=728,
                            out_channels=728,
                            kernel_size=3,
                            padding='same',
                            bias=False,
                            stride=1),
            torch.nn.BatchNorm2d(num_features=728),
            activation(),
            SeparableConv2d(in_channels=728,
                            out_channels=728,
                            kernel_size=3,
                            padding='same',
                            bias=False,
                            stride=1),
            torch.nn.BatchNorm2d(num_features=728),
            activation(),
            SeparableConv2d(in_channels=728,
                            out_channels=728,
                            kernel_size=3,
                            padding='same',
                            bias=False,
                            stride=1),
            torch.nn.BatchNorm2d(num_features=728)
        )

    def forward(self, x):
        y = self.features(x) + x
        return y



class Xception(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.ReLU):
        super(Xception, self).__init__()

        self.class_num = class_num

        #Entry Flow
        #Entry Flow
        self.prelayer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=32,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=32),
            activation(),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding='same',
                            bias=False),
            torch.nn.BatchNorm2d(num_features=64),
            activation()
        )

        #layer1
        self.layer1_skip = torch.nn.Conv2d(in_channels=64,
                                           out_channels=128,
                                           kernel_size=1,
                                           bias=True,
                                           stride=2)


        self.layer1 = torch.nn.Sequential(
            SeparableConv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding='same',
                            bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            activation(),
            SeparableConv2d(in_channels=128,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding='same',
                            bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.MaxPool2d(kernel_size=3,
                               stride=2,
                               padding=1)
        )

        # layer2
        self.layer2_skip = torch.nn.Conv2d(in_channels=128,
                                           out_channels=256,
                                           kernel_size=1,
                                           bias=True,
                                           stride=2)

        self.layer2 = torch.nn.Sequential(
            SeparableConv2d(in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            stride=1,
                            padding='same',
                            bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            activation(),
            SeparableConv2d(in_channels=256,
                            out_channels=256,
                            kernel_size=3,
                            stride=1,
                            padding='same',
                            bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.MaxPool2d(kernel_size=3,
                               stride=2,
                               padding=1)
        )

        # layer3
        self.layer3_skip = torch.nn.Conv2d(in_channels=256,
                                           out_channels=728,
                                           kernel_size=1,
                                           bias=True,
                                           stride=2)

        self.layer3 = torch.nn.Sequential(
            SeparableConv2d(in_channels=256,
                            out_channels=728,
                            kernel_size=3,
                            stride=1,
                            padding='same',
                            bias=False),
            torch.nn.BatchNorm2d(num_features=728),
            activation(),
            SeparableConv2d(in_channels=728,
                            out_channels=728,
                            kernel_size=3,
                            stride=1,
                            padding='same',
                            bias=False),
            torch.nn.BatchNorm2d(num_features=728),
            torch.nn.MaxPool2d(kernel_size=3,
                               stride=2,
                               padding=1)
        )
        #Entry Flow
        #Entry Flow


        #Middle Flow
        #Middle Flow
        self.middle_flow = torch.nn.Sequential(
            MiddleFlow(),
            MiddleFlow(),
            MiddleFlow(),
            MiddleFlow(),
            MiddleFlow(),
            MiddleFlow(),
            MiddleFlow(),
            MiddleFlow(),
        )
        #Middle Flow
        #Middle Flow

        #Exit Flow
        # Exit Flow
        self.exit_flow1_skip = torch.nn.Conv2d(kernel_size=1,
                                               stride=2,
                                               bias=True,
                                               in_channels=728,
                                               out_channels=1024)
        self.exit_flow1 = torch.nn.Sequential(
            activation(),
            SeparableConv2d(kernel_size=3,
                            in_channels=728,
                            out_channels=728,
                            bias=False,
                            stride=1,
                            padding='same'),
            torch.nn.BatchNorm2d(num_features=728),
            activation(),
            SeparableConv2d(kernel_size=3,
                            in_channels=728,
                            out_channels=1024,
                            bias=False,
                            stride=1,
                            padding='same'),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.MaxPool2d(kernel_size=3,
                               stride=2,
                               padding=1)
        )

        self.exit_flow2 = torch.nn.Sequential(
            SeparableConv2d(kernel_size=3,
                            in_channels=1024,
                            out_channels=1536,
                            padding='same',
                            bias=False,
                            stride=1),
            torch.nn.BatchNorm2d(num_features=1536),
            activation(),
            SeparableConv2d(kernel_size=3,
                            in_channels=1536,
                            out_channels=self.class_num,
                            padding='same',
                            bias=True,
                            stride=1),
            torch.nn.AdaptiveAvgPool2d(1),

        )


        #Exit Flow
        #Exit Flow

    def forward(self, x):

        #Entry flow
        x = self.prelayer1(x)
        x = self.layer1(x) + self.layer1_skip(x)
        x = self.layer2(x) + self.layer2_skip(x)
        x = self.layer3(x) + self.layer3_skip(x)
        # Entry flow 19x19x728


        # Middle flow
        x = self.middle_flow(x)
        # Middle flow 19x19x728


        # Exit flow
        x = self.exit_flow1(x) + self.exit_flow1_skip(x)
        x = self.exit_flow2(x)
        # Exit flow

        x = x.view([-1, self.class_num])
        x = torch.sigmoid(x)

        return x