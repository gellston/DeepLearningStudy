import torch
import torch.nn.functional as F
from util.helper import CBAMCSPResidualBlock
from util.helper import ResidualBlock
from util.helper import SeparableActivationConv2d
from util.helper import SESeparableActivationConv2d

class ReagentModelV5(torch.nn.Module):

    def __init__(self, class_num=3, activation=torch.nn.ReLU6):
        super(ReagentModelV5, self).__init__()

        self.class_num = class_num

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=32,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation(),
                                                    # 248x88
            SESeparableActivationConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False, activation=activation, reduction_rate=4),
            SESeparableActivationConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=False, activation=activation, reduction_rate=4),
                                                    # 124x44
            SESeparableActivationConv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False, activation=activation, reduction_rate=4),
            SESeparableActivationConv2d(in_channels=64, out_channels=75, kernel_size=3, padding=1, stride=2, bias=False, activation=activation, reduction_rate=4),
                                                    # 64x22
            SESeparableActivationConv2d(in_channels=75, out_channels=75, kernel_size=3, padding=1, stride=1, bias=False, activation=activation, reduction_rate=4),
            SESeparableActivationConv2d(in_channels=75, out_channels=86, kernel_size=3, padding=1, stride=2, bias=False, activation=activation, reduction_rate=4),
                                                    # 32x11
            SESeparableActivationConv2d(in_channels=86, out_channels=86, kernel_size=3, padding=1, stride=1, bias=False, activation=activation, reduction_rate=4),
            SESeparableActivationConv2d(in_channels=86, out_channels=99, kernel_size=3, padding=1, stride=2, bias=False, activation=activation, reduction_rate=4),
                                                    # 16x5
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Dropout2d(p=0.25),
            torch.nn.Conv2d(in_channels=99,
                            out_channels=99,
                            kernel_size=1,
                            bias=True),
            torch.nn.Dropout2d(p=0.25),
            torch.nn.Conv2d(in_channels=99,
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