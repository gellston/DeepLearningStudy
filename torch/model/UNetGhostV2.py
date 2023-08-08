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

        #144x400
        self.stem1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            bias=False,
                            stride=1,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),
            activation()
        )

        #72x200
        self.stem2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16,
                            out_channels=16,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),
            activation()
        )


        # 36x100
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

        # 18x50
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

        # 9x25
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

        # 9x25
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

        # 18x50
        self.transpose_layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=16,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3)
        )

        # 18x50
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



        # 36x100
        self.transpose_layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=16,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3)
        )

        # 36x100
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

        #72x200
        self.transpose_layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=16,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3)
        )

        #72x200
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



        # 144x400
        self.transpose_layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=16,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3)
        )

        # 144x400
        self.feature_mix_layer5 = torch.nn.Sequential(
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





        self.final_conv1 = torch.nn.Sequential(
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


        stem1 = self.stem1(x)           #144x400
        stem2 = self.stem2(stem1)       #72x200
        layer1 = self.layer1(stem2)     #36x100
        layer2 = self.layer2(layer1)    #18x50
        layer3 = self.layer3(layer2)    #9x25

        latent_space = self.latent_space(layer3) #9x25

        transpose_layer1 = self.transpose_layer1(latent_space)          #18x50
        concat1 = torch.concat([transpose_layer1, layer2], dim=1)
        feature_mix_layer2 = self.feature_mix_layer2(concat1)


        transpose_layer2 = self.transpose_layer2(feature_mix_layer2)    #36x100
        concat2 = torch.concat([transpose_layer2, layer1], dim=1)
        feature_mix_layer3 = self.feature_mix_layer3(concat2)

        transpose_layer3 = self.transpose_layer3(feature_mix_layer3)    #72x200
        concat3 = torch.concat([transpose_layer3, stem2], dim=1)
        feature_mix_layer4 = self.feature_mix_layer4(concat3)

        transpose_layer4 = self.transpose_layer4(feature_mix_layer4)    #144x400
        concat4 = torch.concat([transpose_layer4, stem1], dim=1)
        feature_mix_layer5 = self.feature_mix_layer5(concat4)

        final_conv1 = self.final_conv1(feature_mix_layer5)
        final_conv1 = torch.sigmoid(final_conv1)


        final_conv2 = self.final_conv2(feature_mix_layer4)
        final_conv2 = torch.nn.functional.interpolate(final_conv2, scale_factor=2.0, mode='bilinear')
        final_conv2 = torch.sigmoid(final_conv2)


        final_conv3 = self.final_conv3(feature_mix_layer3)
        final_conv3 = torch.nn.functional.interpolate(final_conv3, scale_factor=4.0, mode='bilinear')
        final_conv3 = torch.sigmoid(final_conv3)

        return final_conv1, final_conv2, final_conv3