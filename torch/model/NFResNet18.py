import torch
from util.helper import ScaledStdConv2d
from util.helper import NFResidualBlock

class NFResNet18(torch.nn.Module):

    def __init__(self,
                 class_num=5):
        super(NFResNet18, self).__init__()

        self.class_num = class_num

        self.layer1 = torch.nn.Sequential(
            ScaledStdConv2d(in_channels=3,
                            out_channels=64,
                            stride=2,
                            padding=3,
                            kernel_size=7),
            torch.nn.MaxPool2d(kernel_size=3,
                               padding=1,
                               stride=2)
        )

        self.layer2 = torch.nn.Sequential(
            NFResidualBlock(in_dim=64,
                            mid_dim=64,
                            out_dim=64,
                            stride=1),
            NFResidualBlock(in_dim=64,
                            mid_dim=64,
                            out_dim=64,
                            stride=1),
        )

        self.layer3 = torch.nn.Sequential(
            NFResidualBlock(in_dim=64,
                            mid_dim=128,
                            out_dim=128,
                            stride=2),
            NFResidualBlock(in_dim=128,
                            mid_dim=128,
                            out_dim=128,
                            stride=1),
        )

        self.layer4 = torch.nn.Sequential(
            NFResidualBlock(in_dim=128,
                            mid_dim=256,
                            out_dim=256,
                            stride=2),
            NFResidualBlock(in_dim=256,
                            mid_dim=256,
                            out_dim=256,
                            stride=1),
        )
        self.layer5 = torch.nn.Sequential(
            NFResidualBlock(in_dim=256,
                            mid_dim=512,
                            out_dim=512,
                            stride=2),
            NFResidualBlock(in_dim=512,
                            mid_dim=512,
                            out_dim=512,
                            stride=1),
        )
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Conv2d(kernel_size=1,
                                  in_channels=512,
                                  out_channels=class_num)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        """ 공부 필요
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)
            공부 필요
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x