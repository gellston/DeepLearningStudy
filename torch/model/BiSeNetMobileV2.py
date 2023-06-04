import torch
import torch.nn.functional as F


## Backbone Helper
from util.helper import InvertedBottleNeck
## BiSeNet Helper
from util.BiSeNetHelper import Spatial_path
from util.BiSeNetHelper import AttentionRefinementModule
from util.BiSeNetHelper import FeatureFusionModule


class BiSeNetMobileV2(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.ReLU6):
        super(BiSeNetMobileV2, self).__init__()

        self.class_num = class_num


        ##Back bone
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),
            activation(),
        ) #2

        self.layer1 = torch.nn.Sequential(
            InvertedBottleNeck(in_channels=16, out_channels=16, expansion_rate=1, stride=1, activation=activation),
            InvertedBottleNeck(in_channels=16, out_channels=24, expansion_rate=6, stride=2, activation=activation)
        ) #4

        self.layer2 = torch.nn.Sequential(
            InvertedBottleNeck(in_channels=24, out_channels=24, expansion_rate=6, stride=1, activation=activation),
            InvertedBottleNeck(in_channels=24, out_channels=32, expansion_rate=6, stride=2, activation=activation)
        ) #8

        self.layer3 = torch.nn.Sequential(
            InvertedBottleNeck(in_channels=32, out_channels=40, expansion_rate=6, stride=1, activation=activation),
            InvertedBottleNeck(in_channels=40, out_channels=40, expansion_rate=6, stride=1, activation=activation),
            InvertedBottleNeck(in_channels=40, out_channels=40, expansion_rate=6, stride=2, activation=activation)
        ) #16

        ##Back bone


        ##BiSeNet
        self.spatial_path = Spatial_path()

        self.attention_refinement_module1 = AttentionRefinementModule(32, 32)
        self.attention_refinement_module2 = AttentionRefinementModule(40, 40)
        self.global_avg_pool_tail = torch.nn.AdaptiveAvgPool2d(1)

        self.feature_fusion_module = FeatureFusionModule(num_classes=self.class_num, in_channels=48 + 32 + 40)
        self.final_conv = torch.nn.Conv2d(in_channels=self.class_num,
                                          out_channels=self.class_num,
                                          kernel_size=3,
                                          padding='same')
        ##BiSeNet



        ##Super vision
        self.super_vision1 = torch.nn.Conv2d(in_channels=32, out_channels=self.class_num, kernel_size=1)
        self.super_vision2 = torch.nn.Conv2d(in_channels=40, out_channels=self.class_num, kernel_size=1)
        ##Super vision


        # module 초기화
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()

    def forward(self, x):

        # spatial path
        spatial_path = self.spatial_path(x)  # x8
        # spatial path

        #context path
                                           #x =>      1,160,416 X1
        stem = self.stem(x)                #stem =>   1,80,208  X2
        layer1 = self.layer1(stem)         #layer1 => 1,40,104  X4
        layer2 = self.layer2(layer1)       #layer2 => 1,20,52   X8
        layer3 = self.layer3(layer2)       #layer3 => 1,10,26   X16


        arm1 = self.attention_refinement_module1(layer2)
        arm2 = self.attention_refinement_module2(layer3)
        tail = self.global_avg_pool_tail(layer3)
        arm2 = torch.mul(arm2, tail)

        arm1 = torch.nn.functional.interpolate(arm1, size=spatial_path.size()[-2:], mode='bilinear')
        arm2 = torch.nn.functional.interpolate(arm2, size=spatial_path.size()[-2:], mode='bilinear')
        context_path = torch.cat((arm1, arm2), dim=1)
        #context path


        #Feature Fusion Module
        ffm = self.feature_fusion_module(spatial_path, context_path)
        # Feature Fusion Module

        #Super vision path
        spv1 = self.super_vision1(arm1)
        spv1 = torch.nn.functional.interpolate(spv1, size=x.size()[-2:], mode='bilinear')
        spv1 = torch.sigmoid(spv1)

        spv2 = self.super_vision2(arm2)
        spv2 = torch.nn.functional.interpolate(spv2, size=x.size()[-2:], mode='bilinear')
        spv2 = torch.sigmoid(spv2)

        #Super vision path

        result = torch.nn.functional.interpolate(ffm, scale_factor=8.0, mode='bilinear')
        result = self.final_conv(result)
        result = torch.sigmoid(result)

        return result, spv1, spv2