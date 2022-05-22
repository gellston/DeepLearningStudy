import torch
import torch.nn.functional as F
from util.helper import CSPInvertedBottleNect


class CSPMobileNetV2(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.ReLU6):
        super(CSPMobileNetV2, self).__init__()

        self.class_num = class_num

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=32,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation(),
            CSPInvertedBottleNect(in_channels=32, out_channels=16, expansion_rate=1, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=16, out_channels=24, expansion_rate=6, stride=2, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=24, out_channels=24, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=24, out_channels=32, expansion_rate=6, stride=2, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=32, out_channels=32, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=32, out_channels=32, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=32, out_channels=64, expansion_rate=6, stride=2, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=64, out_channels=64, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=64, out_channels=64, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=64, out_channels=64, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=64, out_channels=96, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=96, out_channels=96, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=96, out_channels=96, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=96, out_channels=160, expansion_rate=6, stride=2, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=160, out_channels=160, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=160, out_channels=160, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            CSPInvertedBottleNect(in_channels=160, out_channels=320, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.ReLU6),
            torch.nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, bias=True),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels=1280,
                            out_channels=self.class_num,
                            kernel_size=1,
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

            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.features(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x