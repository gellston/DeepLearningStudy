import torch
import torch.nn.functional as F
from util.helper import InvertedBottleNeck


class MobileNetV2(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.ReLU6):
        super(MobileNetV2, self).__init__()

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
            InvertedBottleNeck(in_channels=32, out_channels=16, expansion_rate=1, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=16, out_channels=24, expansion_rate=6, stride=2, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=24, out_channels=24, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=24, out_channels=32, expansion_rate=6, stride=2, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=32, out_channels=32, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=32, out_channels=32, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=32, out_channels=64, expansion_rate=6, stride=2, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=64, out_channels=64, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=64, out_channels=64, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=64, out_channels=64, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=64, out_channels=96, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=96, out_channels=96, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=96, out_channels=96, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=96, out_channels=160, expansion_rate=6, stride=2, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=160, out_channels=160, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=160, out_channels=160, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            InvertedBottleNeck(in_channels=160, out_channels=320, expansion_rate=6, stride=1, activation=torch.nn.ReLU6),
            torch.nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, bias=False),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels=1280,
                            out_channels=self.class_num,
                            kernel_size=1,
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