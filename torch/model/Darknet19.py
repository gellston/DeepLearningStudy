import torch
from util.helper import CSPResidualBlock

class Darknet19(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.ReLU):
        super(Darknet19, self).__init__()
        self.class_num = class_num

        self.features = torch.nn.Sequential(

            # Layer1
            torch.nn.Conv2d(in_channels=3,
                            out_channels=32,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            #Layer2
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            #Layer3
            torch.nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=64,
                            kernel_size=1,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # Layer4
            torch.nn.Conv2d(in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=128,
                            kernel_size=1,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            #Layer5
            torch.nn.Conv2d(in_channels=256,
                            out_channels=512,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=256,
                            kernel_size=1,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=512,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=256,
                            kernel_size=1,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=512,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # Layer6
            torch.nn.Conv2d(in_channels=512,
                            out_channels=1024,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=1024,
                            out_channels=512,
                            kernel_size=1,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=1024,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=1024,
                            out_channels=512,
                            kernel_size=1,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=1024,
                            kernel_size=3,
                            bias=False,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.1),
        )

        self.final_conv = torch.nn.Conv2d(kernel_size=1,
                                          in_channels=1024,
                                          out_channels=self.class_num,
                                          bias=True)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

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
        x = self.final_conv(x)
        x = self.avg_pool(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x