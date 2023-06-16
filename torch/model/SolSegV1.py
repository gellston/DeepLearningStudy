import torch
from util.helper import SelfAttResBlock
from util.helper import SEConvBlock

class SolSegV1(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.LeakyReLU):
        super(SolSegV1, self).__init__()

        self.class_num = class_num


        ##Back bone
        # 256x64
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=8,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(8),
            activation(),
            SEConvBlock(in_channels=8,
                        channels=8,
                        se_rate=2)
        )
        #128x32
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8,
                            out_channels=16,
                            groups=2,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),
            activation(),
            SEConvBlock(in_channels=16,
                        channels=16,
                        se_rate=4)
        )
        #64x16
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16,
                            out_channels=32,
                            groups=2,
                            bias=False,
                            stride=2,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation(),
            SEConvBlock(in_channels=32,
                        channels=32,
                        se_rate=4)
        )



        self.resize_conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8,
                            out_channels=32,
                            groups=2,
                            bias=False,
                            stride=1,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation(),
            SEConvBlock(in_channels=32,
                        channels=32,
                        se_rate=4)
        )

        self.resize_conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16,
                            out_channels=32,
                            groups=2,
                            bias=False,
                            stride=1,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation(),
            SEConvBlock(in_channels=32,
                        channels=32,
                        se_rate=4)
        )

        self.resize_conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=32,
                            groups=2,
                            bias=False,
                            stride=1,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(32),
            activation(),
            SEConvBlock(in_channels=32,
                        channels=32,
                        se_rate=4)
        )

        self.final_conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=24,
                            bias=False,
                            stride=1,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(24),
            activation(),
            SEConvBlock(in_channels=24,
                        channels=24,
                        se_rate=4),
            torch.nn.Conv2d(in_channels=24,
                            out_channels=16,
                            bias=False,
                            stride=1,
                            padding=1,
                            kernel_size=3),
            torch.nn.BatchNorm2d(16),
            activation(),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=self.class_num,
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

        layer1 = self.layer1(x)      #256x64x8
        layer2 = self.layer2(layer1) #128x32x16
        layer3 = self.layer3(layer2) #64x16x32



        up_layer2 = torch.nn.functional.interpolate(layer2, size=layer1.size()[-2:], mode='bilinear')
        up_layer3 = torch.nn.functional.interpolate(layer3, size=layer1.size()[-2:], mode='bilinear')


        resize_conv3 = self.resize_conv3(up_layer3)
        resize_conv2 = self.resize_conv2(up_layer2) + resize_conv3
        resize_conv1 = self.resize_conv1(layer1) + resize_conv2

        final = torch.nn.functional.interpolate(resize_conv1, size=x.size()[-2:], mode='bilinear')
        final = self.final_conv1(final)
        final = torch.sigmoid(final)


        return final