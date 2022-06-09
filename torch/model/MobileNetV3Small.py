import torch
from util.helper import InvertedBottleNectV3


class MobileNetV3Small(torch.nn.Module):

    def __init__(self, class_num=5):
        super(MobileNetV3Small, self).__init__()

        self.class_num = class_num

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=16,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.Hardswish(),
            InvertedBottleNectV3(in_channels=16, expansion_out=16, out_channels=16, stride=2, use_se=True, #200x200 3
                                 kernel_size=3,
                                 activation=torch.nn.ReLU),

            InvertedBottleNectV3(in_channels=16, expansion_out=72, out_channels=24, stride=2, use_se=False,
                                 kernel_size=3,
                                 activation=torch.nn.ReLU),

            InvertedBottleNectV3(in_channels=24, expansion_out=88, out_channels=24, stride=1, use_se=False, #100x100 5
                                 kernel_size=3,
                                 activation=torch.nn.ReLU),

            InvertedBottleNectV3(in_channels=24, expansion_out=96, out_channels=40, stride=2, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=40, expansion_out=240, out_channels=40, stride=1, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=40, expansion_out=240, out_channels=40, stride=1, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=40, expansion_out=120, out_channels=48, stride=1, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=48, expansion_out=144, out_channels=48, stride=1, use_se=True, #50x50 10
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=48, expansion_out=288, out_channels=96, stride=2, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=96, expansion_out=576, out_channels=96, stride=1, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=96, expansion_out=576, out_channels=96, stride=1, use_se=True, #20x20 13
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            torch.nn.Conv2d(kernel_size=1,
                            bias=False,
                            in_channels=96,
                            out_channels=576),
            torch.nn.BatchNorm2d(num_features=576),
            torch.nn.Hardswish(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=576,
                            out_channels=1024,
                            bias=True),
            torch.nn.Hardswish(),
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=1024,
                            out_channels=self.class_num,
                            bias=True)
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


    def forward(self, x):
        x = self.features(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x