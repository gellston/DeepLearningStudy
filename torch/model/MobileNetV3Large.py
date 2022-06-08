import torch
from util.helper import InvertedBottleNectV3


class MobileNetV3Large(torch.nn.Module):

    def __init__(self, class_num=5):
        super(MobileNetV3Large, self).__init__()

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
            InvertedBottleNectV3(in_channels=16, expansion_out=16, out_channels=16, stride=1, use_se=False,
                                 kernel_size=3,
                                 activation=torch.nn.ReLU),

            InvertedBottleNectV3(in_channels=16, expansion_out=64, out_channels=24, stride=2, use_se=False,
                                 kernel_size=3,
                                 activation=torch.nn.ReLU),

            InvertedBottleNectV3(in_channels=24, expansion_out=72, out_channels=24, stride=1, use_se=False,
                                 kernel_size=3,
                                 activation=torch.nn.ReLU),

            InvertedBottleNectV3(in_channels=24, expansion_out=72, out_channels=40, stride=2, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=40, expansion_out=120, out_channels=40, stride=1, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=40, expansion_out=120, out_channels=40, stride=1, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=40, expansion_out=240, out_channels=80, stride=2, use_se=False,
                                 kernel_size=3,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=80, expansion_out=200, out_channels=80, stride=1, use_se=False,
                                 kernel_size=3,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=80, expansion_out=184, out_channels=80, stride=1, use_se=False,
                                 kernel_size=3,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=80, expansion_out=184, out_channels=80, stride=1, use_se=False,
                                 kernel_size=3,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=80, expansion_out=480, out_channels=112, stride=1, use_se=True,
                                 kernel_size=3,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=112, expansion_out=672, out_channels=112, stride=1, use_se=True,
                                 kernel_size=3,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=112, expansion_out=672, out_channels=160, stride=2, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=160, expansion_out=960, out_channels=160, stride=1, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            InvertedBottleNectV3(in_channels=160, expansion_out=960, out_channels=160, stride=1, use_se=True,
                                 kernel_size=5,
                                 activation=torch.nn.Hardswish),

            torch.nn.Conv2d(kernel_size=1,
                            bias=False,
                            in_channels=160,
                            out_channels=960),
            torch.nn.BatchNorm2d(num_features=960),
            torch.nn.Hardswish(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=960,
                            out_channels=1280,
                            bias=True),
            torch.nn.Hardswish(),
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=1280,
                            out_channels=self.class_num,
                            bias=True)
        )

        # module 초기화
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x