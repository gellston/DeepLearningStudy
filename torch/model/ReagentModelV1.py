import torch
import torch.nn.functional as F
from util.helper import CBAM
from util.helper import CSPResidualBlock


class ReagentModelV1(torch.nn.Module):

    def __init__(self, class_num=3, activation=torch.nn.ReLU6):
        super(ReagentModelV1, self).__init__()

        self.class_num = class_num

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),
            activation(),                          #256x144
            torch.nn.Conv2d(in_channels=16,
                            out_channels=32,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation(),                           #128x72
            CSPResidualBlock(in_dim=32, mid_dim=32, out_dim=32, stride=1, activation=activation),
            CSPResidualBlock(in_dim=32, mid_dim=44, out_dim=44, stride=2, activation=activation),
                                                    # 64x36

            CSPResidualBlock(in_dim=44, mid_dim=44, out_dim=44, stride=1, activation=activation),
            CSPResidualBlock(in_dim=44, mid_dim=55, out_dim=55, stride=2, activation=activation),
                                                    # 32x18

            CSPResidualBlock(in_dim=55, mid_dim=55, out_dim=55, stride=1, activation=activation),
            CSPResidualBlock(in_dim=55, mid_dim=66, out_dim=66, stride=2, activation=activation),
                                                    # 16x9

            torch.nn.Conv2d(in_channels=66, out_channels=77, kernel_size=1, stride=1, bias=False),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels=77,
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