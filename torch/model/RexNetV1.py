import torch
from util.helper import RexNetLinearBottleNeck


class RexNetV1(torch.nn.Module):

    def __init__(self,
                 class_num=5,
                 width_multple=1.0,
                 se_rate=12,
                 dropout_ratio=0.2):
        super(RexNetV1, self).__init__()

        self.class_num = class_num
        self.stem_channel = int(32 * width_multple) if width_multple < 1.0 else 32
        self.width_multiple = width_multple
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=self.stem_channel,
                            kernel_size=3,
                            bias=False,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=self.stem_channel),
            torch.nn.SiLU(),
            RexNetLinearBottleNeck(in_channels=self.stem_channel,
                                   out_channels=int(16 * self.width_multiple),
                                   stride=1,
                                   use_se=False,
                                   expand_rate=1,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(16 * self.width_multiple),
                                   out_channels=int(27 * self.width_multiple),
                                   stride=2,
                                   use_se=False,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(27 * self.width_multiple),
                                   out_channels=int(38 * self.width_multiple),
                                   stride=1,
                                   use_se=False,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(38 * self.width_multiple),
                                   out_channels=int(50 * self.width_multiple),
                                   stride=2,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(50 * self.width_multiple),
                                   out_channels=int(61 * self.width_multiple),
                                   stride=1,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(61 * self.width_multiple),
                                   out_channels=int(72 * self.width_multiple),
                                   stride=2,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(72 * self.width_multiple),
                                   out_channels=int(84 * self.width_multiple),
                                   stride=1,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(84 * self.width_multiple),
                                   out_channels=int(95 * self.width_multiple),
                                   stride=1,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(95 * self.width_multiple),
                                   out_channels=int(106 * self.width_multiple),
                                   stride=1,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(106 * self.width_multiple),
                                   out_channels=int(117 * self.width_multiple),
                                   stride=1,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(117 * self.width_multiple),
                                   out_channels=int(128 * self.width_multiple),
                                   stride=1,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(128 * self.width_multiple),
                                   out_channels=int(140 * self.width_multiple),
                                   stride=2,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(140 * self.width_multiple),
                                   out_channels=int(151 * self.width_multiple),
                                   stride=1,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(151 * self.width_multiple),
                                   out_channels=int(162 * self.width_multiple),
                                   stride=1,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(162 * self.width_multiple),
                                   out_channels=int(174 * self.width_multiple),
                                   stride=1,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
            RexNetLinearBottleNeck(in_channels=int(174 * self.width_multiple),
                                   out_channels=int(185 * self.width_multiple),
                                   stride=1,
                                   use_se=True,
                                   expand_rate=6,
                                   se_rate=se_rate),
        )

        self.final_conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=int(185 * self.width_multiple),
                            out_channels=int(1280 * self.width_multiple),
                            bias=False),
            torch.nn.BatchNorm2d(num_features=int(1280 * self.width_multiple)),
            torch.nn.SiLU()
        )
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.final_conv2 = torch.nn.Sequential(
            torch.nn.Dropout2d(p=dropout_ratio),
            torch.nn.Conv2d(kernel_size=1,
                            in_channels=int(1280 * self.width_multiple),
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
        x = self.layers(x)
        x = self.final_conv1(x)
        x = self.global_avg_pool(x)
        x = self.final_conv2(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x