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
                            out_channels=32,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation(),                           # 248x88


            CSPResidualBlock(in_dim=32, mid_dim=32, out_dim=32, stride=1, activation=activation),
            CSPResidualBlock(in_dim=32, mid_dim=32, out_dim=32, stride=2, activation=activation),
                                                    # 124x44
            CSPResidualBlock(in_dim=32, mid_dim=32, out_dim=32, stride=1, activation=activation),
            CSPResidualBlock(in_dim=32, mid_dim=32, out_dim=32, stride=1, activation=activation),
            CSPResidualBlock(in_dim=32, mid_dim=32, out_dim=32, stride=1, activation=activation),
            CSPResidualBlock(in_dim=32, mid_dim=32, out_dim=32, stride=2, activation=activation),
                                                    # 64x22

            CSPResidualBlock(in_dim=32, mid_dim=32, out_dim=32, stride=1, activation=activation),
            CSPResidualBlock(in_dim=32, mid_dim=32, out_dim=32, stride=2, activation=activation),
                                                    # 32x11
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Dropout2d(p=0.25),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=16,
                            kernel_size=1,
                            bias=True),
            activation(),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=8,
                            kernel_size=1,
                            bias=True),
            activation(),
            torch.nn.Conv2d(in_channels=8,
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