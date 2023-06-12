import torch
import torch.nn.functional as F


class ParticleDetectorV1(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.LeakyReLU):
        super(ParticleDetectorV1, self).__init__()

        self.class_num = class_num


        ##Back bone
        # 368x168
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=8,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(8),
            activation(),


        )
        #168x84
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8,
                            out_channels=16,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),
            activation()
        )
        #84x42
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16,
                            out_channels=32,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation()
        )
        #42x21
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(64),
            activation()
        )

        # 84x42
        self.transpose_conv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64,
                                     out_channels=32,
                                     bias=False,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation()
        )

        # 168x84
        self.transpose_conv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32,
                                     out_channels=16,
                                     bias=False,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3),
            torch.nn.BatchNorm2d(16),
            activation()
        )
        # 368x168
        self.transpose_conv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=16,
                                     out_channels=8,
                                     bias=False,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3),
            torch.nn.BatchNorm2d(8),
            activation()
        )

        self.final_trans_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=8,
                                     out_channels=1,
                                     bias=False,
                                     stride=2,
                                     padding=1,
                                     output_padding=1,
                                     kernel_size=3),
            torch.nn.BatchNorm2d(1),
            activation()
        )

        self.final_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=1,
                            bias=False,
                            stride=1,
                            padding=1,
                            kernel_size=3),
        )


        # module 초기화
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()

    def forward(self, x):

        layer1 = self.layer1(x)      #368x168x8
        layer2 = self.layer2(layer1) #168x84x16
        layer3 = self.layer3(layer2) #84x42x32
        layer4 = self.layer4(layer3) #42x21x64

        trans_layer4 = self.transpose_conv4(layer4) #84x42x32
        trans_layer4 = trans_layer4 + layer3
        trans_layer3 = self.transpose_conv3(trans_layer4) #168x84x16
        trans_layer3 = trans_layer3 + layer2
        trans_layer2 = self.transpose_conv2(trans_layer3) #368x168x8
        trans_layer2 = trans_layer2 + layer1
        final = self.final_trans_conv(trans_layer2)
        final = self.final_conv(final)
        final = torch.sigmoid(final)

        return final