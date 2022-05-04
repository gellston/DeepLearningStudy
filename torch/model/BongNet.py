import torch
import torch.nn.functional as F
from util.helper import CSPInvertedBottleNect


class BongNet(torch.nn.Module):

    def __init__(self, class_num=2, activation=torch.nn.Hardswish):
        super(BongNet, self).__init__()

        self.class_num = class_num

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=3,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(3),
            activation(),
            CSPInvertedBottleNect(in_channels=3, out_channels=3, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.Hardswish),
            CSPInvertedBottleNect(in_channels=3, out_channels=3, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.Hardswish),
            CSPInvertedBottleNect(in_channels=3, out_channels=6, expansion_rate=6, stride=2, part_ratio=0.5, activation=torch.nn.Hardswish),
            CSPInvertedBottleNect(in_channels=6, out_channels=6, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.Hardswish),
            CSPInvertedBottleNect(in_channels=6, out_channels=10, expansion_rate=6, stride=2, part_ratio=0.5, activation=torch.nn.Hardswish),
            CSPInvertedBottleNect(in_channels=10, out_channels=10, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.Hardswish),
            CSPInvertedBottleNect(in_channels=10, out_channels=14, expansion_rate=6, stride=2, part_ratio=0.5, activation=torch.nn.Hardswish),
            CSPInvertedBottleNect(in_channels=14, out_channels=14, expansion_rate=6, stride=1, part_ratio=0.5, activation=torch.nn.Hardswish),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels=14,
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