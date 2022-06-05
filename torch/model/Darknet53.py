import torch
from util.helper import DarknetResidualBlock

class Darknet53(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.LeakyReLU):
        super(Darknet53, self).__init__()
        self.class_num = class_num

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3,
                            in_channels=3,
                            out_channels=32,
                            stride=1,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=32),
            activation(),
            torch.nn.Conv2d(kernel_size=3,
                            in_channels=32,
                            out_channels=64,
                            stride=2,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=64),
            activation(),
            #Residual 32->64x1
            DarknetResidualBlock(in_channels=64,
                                 out_channels=32,
                                 stride=1,
                                 activation=activation),


            torch.nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            activation(),

            #Residual 64->128x2
            DarknetResidualBlock(in_channels=128,
                                 out_channels=64,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=128,
                                 out_channels=64,
                                 stride=1,
                                 activation=activation),

            torch.nn.Conv2d(in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            activation(),

            # Residual 128->256x8
            DarknetResidualBlock(in_channels=256,
                                 out_channels=128,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=256,
                                 out_channels=128,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=256,
                                 out_channels=128,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=256,
                                 out_channels=128,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=256,
                                 out_channels=128,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=256,
                                 out_channels=128,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=256,
                                 out_channels=128,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=256,
                                 out_channels=128,
                                 stride=1,
                                 activation=activation),

            torch.nn.Conv2d(in_channels=256,
                            out_channels=512,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            activation(),

            # Residual 256->512x8
            DarknetResidualBlock(in_channels=512,
                                 out_channels=256,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=512,
                                 out_channels=256,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=512,
                                 out_channels=256,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=512,
                                 out_channels=256,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=512,
                                 out_channels=256,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=512,
                                 out_channels=256,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=512,
                                 out_channels=256,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=512,
                                 out_channels=256,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=512,
                                 out_channels=256,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=512,
                                 out_channels=256,
                                 stride=1,
                                 activation=activation),

            torch.nn.Conv2d(in_channels=512,
                            out_channels=1024,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            activation(),

            # Residual 512->1024x4
            DarknetResidualBlock(in_channels=1024,
                                 out_channels=512,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=1024,
                                 out_channels=512,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=1024,
                                 out_channels=512,
                                 stride=1,
                                 activation=activation),
            DarknetResidualBlock(in_channels=1024,
                                 out_channels=512,
                                 stride=1,
                                 activation=activation),

            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(kernel_size=1,
                            bias=True,
                            in_channels=1024,
                            out_channels=self.class_num)

        )

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
        x = self.features(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x