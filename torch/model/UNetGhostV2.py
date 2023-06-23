import torch
import torch.nn.functional as F


## Backbone Helper
from util.helper import GhostModule
from util.helper import InvertedGhostBottleNeck
## BiSeNet Helper
from util.BiSeNetHelper import spatial_path
from util.BiSeNetHelper import ARM
from util.BiSeNetHelper import FFM


class UNetGhostV2(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.ReLU, use_activation=True, expansion_rate=2):
        super(UNetGhostV2, self).__init__()

        self.class_num = class_num


        ##Back bone

        #512x128
        self.stem1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=8,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(8),
            activation()
        )

        # 256x64
        self.stem2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8,
                            out_channels=16,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),
            activation()
        )

        # 128x32
        self.layer1 = torch.nn.Sequential(
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation),
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=2,
                                    activation=activation)
        )

        # 64x16
        self.layer2 = torch.nn.Sequential(
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation),
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=2,
                                    activation=activation)
        )

        # 32x8
        self.layer3 = torch.nn.Sequential(
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation),
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=2,
                                    activation=activation)
        )

        # 32x8
        self.latent_space = torch.nn.Sequential(
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation),
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation),
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation)
        )

        # 64x16
        self.transpose_layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=16,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3)
        )

        # 64x16
        self.feature_mix_layer2 = torch.nn.Sequential(
            InvertedGhostBottleNeck(in_channels=32,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation),
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation)
        )



        # 128x32
        self.transpose_layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=16,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3)
        )

        # 128x32
        self.feature_mix_layer3 = torch.nn.Sequential(
            InvertedGhostBottleNeck(in_channels=32,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation),
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation)
        )

        # 256x64
        self.transpose_layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=16,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3)
        )

        # 256x64
        self.feature_mix_layer4 = torch.nn.Sequential(
            InvertedGhostBottleNeck(in_channels=32,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation),
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation)
        )



        # 512x128
        self.transpose_layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=16,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3)
        )

        # 512x128
        self.feature_mix_layer5 = torch.nn.Sequential(
            InvertedGhostBottleNeck(in_channels=24,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation),
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation)
        )

        # 1024x256
        self.transpose_layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=16,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3)
        )

        # 1024x256
        self.feature_mix_layer6 = torch.nn.Sequential(
            InvertedGhostBottleNeck(in_channels=24,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation),
            InvertedGhostBottleNeck(in_channels=16,
                                    out_channels=16,
                                    expansion_rate=expansion_rate,
                                    stride=1,
                                    activation=activation)
        )




        self.final_conv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=16,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3,
                                     bias=False),
            torch.nn.BatchNorm2d(num_features=16),
            activation(inplace=True),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=self.class_num,
                            stride=1,
                            padding=1,
                            kernel_size=3,
                            bias=False)
        )


        self.final_conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16,
                            out_channels=16,
                            stride=1,
                            padding=1,
                            kernel_size=3,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=16),
            activation(inplace=True),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=self.class_num,
                            stride=1,
                            padding=1,
                            kernel_size=3,
                            bias=False)
        )

        self.final_conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16,
                            out_channels=16,
                            stride=1,
                            padding=1,
                            kernel_size=3,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=16),
            activation(inplace=True),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=self.class_num,
                            stride=1,
                            padding=1,
                            kernel_size=3,
                            bias=False)
        )

    def forward(self, x):


        stem1 = self.stem1(x)           #512x128
        stem2 = self.stem2(stem1)       #256x64

        layer1 = self.layer1(stem2)     #128x32
        layer2 = self.layer2(layer1)    #64x16
        layer3 = self.layer3(layer2)    #32x8

        latent_space = self.latent_space(layer3) #32x8

        transpose_layer1 = self.transpose_layer1(latent_space)          #64x16
        concat1 = torch.concat([transpose_layer1, layer2], dim=1)
        feature_mix_layer2 = self.feature_mix_layer2(concat1)


        transpose_layer2 = self.transpose_layer2(feature_mix_layer2)    #128x32
        concat2 = torch.concat([transpose_layer2, layer1], dim=1)
        feature_mix_layer3 = self.feature_mix_layer3(concat2)

        transpose_layer3 = self.transpose_layer3(feature_mix_layer3)    #256x64
        concat3 = torch.concat([transpose_layer3, stem2], dim=1)
        feature_mix_layer4 = self.feature_mix_layer4(concat3)

        transpose_layer4 = self.transpose_layer4(feature_mix_layer4)    #512x128
        concat4 = torch.concat([transpose_layer4, stem1], dim=1)
        feature_mix_layer5 = self.feature_mix_layer5(concat4)

        final_conv1 = self.final_conv1(feature_mix_layer5)
        final_conv1 = torch.sigmoid(final_conv1)


        final_conv2 = self.final_conv2(feature_mix_layer4)
        final_conv2 = torch.nn.functional.interpolate(final_conv2, scale_factor=4.0, mode='bilinear')
        final_conv2 = torch.sigmoid(final_conv2)


        final_conv3 = self.final_conv3(feature_mix_layer3)
        final_conv3 = torch.nn.functional.interpolate(final_conv3, scale_factor=8.0, mode='bilinear')
        final_conv3 = torch.sigmoid(final_conv3)

        return final_conv1, final_conv2, final_conv3