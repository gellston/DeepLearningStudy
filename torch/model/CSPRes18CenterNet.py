import torch
import torch.nn.functional as F
from util.helper import CSPResidualBlock


class CSPRes18CenterNet(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.SiLU):
        super(CSPRes18CenterNet, self).__init__()

        self.class_num = class_num
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3,
                                                         out_channels=64,
                                                         kernel_size=3,
                                                         stride=2,
                                                         padding=1,
                                                         bias=False),
                                         torch.nn.BatchNorm2d(num_features=64),
                                         activation())

        self.conv2 = torch.nn.Sequential(CSPResidualBlock(in_dim=64, mid_dim=64, out_dim=64, stride=1,
                                                          activation=activation),
                                         CSPResidualBlock(in_dim=64, mid_dim=64, out_dim=128, stride=2,
                                                          activation=activation))

        self.conv3 = torch.nn.Sequential(CSPResidualBlock(in_dim=128, mid_dim=128, out_dim=128, stride=1,
                                                          activation=activation),
                                         CSPResidualBlock(in_dim=128, mid_dim=128, out_dim=256, stride=2,
                                                          activation=activation))

        self.conv4 = torch.nn.Sequential(CSPResidualBlock(in_dim=256, mid_dim=256, out_dim=256, stride=1,
                                                          activation=activation),
                                         CSPResidualBlock(in_dim=256, mid_dim=256, out_dim=512, stride=2,
                                                          activation=activation))

        self.conv5 = torch.nn.Sequential(CSPResidualBlock(in_dim=512, mid_dim=512, out_dim=512, stride=1,
                                                          activation=activation),
                                         CSPResidualBlock(in_dim=512, mid_dim=512, out_dim=512, stride=2,
                                                          activation=activation))

        self.global_average_pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.final_conv = torch.nn.Conv2d(in_channels=512,
                                          out_channels=self.class_num,
                                          kernel_size=1,
                                          bias=True,
                                          padding='same')
        self.sigmoid = torch.nn.Sigmoid()



        ##Feature Pyramid Network
        self.feature_extraction1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=512,
                                                                       out_channels=128,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(128),
                                                       activation())  # 16x16

        self.feature_extraction2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=512,
                                                                       out_channels=128,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(128),
                                                       activation()) #32x32

        self.feature_extraction3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=256,
                                                                       out_channels=128,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(128),
                                                       activation()) #64x64

        self.feature_extraction4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128,
                                                                       out_channels=128,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(128),
                                                       activation()) #128x128

        """
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=512,
                                                out_channels=128,
                                                kernel_size=2,
                                                bias=False,
                                                stride=2) #28x28

        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=128,
                                                out_channels=128,
                                                kernel_size=2,
                                                bias=False,
                                                stride=2) #56x56

        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=128,
                                                out_channels=128,
                                                kernel_size=2,
                                                bias=False,
                                                stride=2) #112x112
        """

        self.up_sample1 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_sample2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_sample3 = torch.nn.UpsamplingBilinear2d(scale_factor=2)


        self.feature_final_conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=256,
                                                                       out_channels=128,
                                                                       kernel_size=3,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(128),
                                                       activation()) #16x16


        self.feature_final_conv2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=256,
                                                                       out_channels=128,
                                                                       kernel_size=3,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(128),
                                                       activation()) #32x32

        self.feature_final_conv3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=256,
                                                                       out_channels=128,
                                                                       kernel_size=3,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(128),
                                                       activation()) #64x64


        ##Feature Pyramid Network


        ##PredictionMap
        self.class_heatmap = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128,
                                                                 out_channels=64,
                                                                 kernel_size=3,
                                                                 bias=False,
                                                                 padding='same'),
                                                 torch.nn.BatchNorm2d(64),
                                                 activation(),
                                                 torch.nn.Conv2d(in_channels=64,
                                                                 out_channels=1,
                                                                 kernel_size=1,
                                                                 bias=True,
                                                                 padding='same'))

        self.size_map = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128,
                                                            out_channels=64,
                                                            kernel_size=3,
                                                            bias=False,
                                                            padding='same'),
                                            torch.nn.BatchNorm2d(64),
                                            activation(),
                                            torch.nn.Conv2d(in_channels=64,
                                                            out_channels=2,
                                                            kernel_size=1,
                                                            bias=True,
                                                            padding='same'),
                                            torch.nn.ReLU())

        self.offset_map = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128,
                                                              out_channels=64,
                                                              kernel_size=3,
                                                              bias=False,
                                                              padding='same'),
                                              torch.nn.BatchNorm2d(64),
                                              activation(),
                                              torch.nn.Conv2d(in_channels=64,
                                                              out_channels=2,
                                                              kernel_size=1,
                                                              bias=True,
                                                              padding='same'),
                                              torch.nn.ReLU())

        ##PredictionMap


        # module 초기화
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()



    def forward(self, x):

        ##Classificaiton
        conv1 = self.conv1(x)       #256x256x64
        conv2 = self.conv2(conv1)   #128x128x128
        conv3 = self.conv3(conv2)   #64x64x256
        conv4 = self.conv4(conv3)   #32x32x512
        conv5 = self.conv5(conv4)   #16x16x512

        x = self.global_average_pooling(conv5)
        x = self.final_conv(x)
        x = x.view([-1, self.class_num])
        classificaiton = self.sigmoid(x)
        ##Classificaiton


        ##Feature Pyramid

        feature1 = self.feature_extraction1(conv5)      #16x16x128
        feature2 = self.feature_extraction2(conv4)      #32x32x128
        feature3 = self.feature_extraction3(conv3)      #64x64x128
        feature4 = self.feature_extraction4(conv2)      #128x128x128

        # 32x32x128
        up_sample1 = torch.cat([self.up_sample1(feature1), feature2], dim=1)
        final_feature1 = self.feature_final_conv1(up_sample1)
        # 64x64x128
        up_sample2 = torch.cat([self.up_sample1(final_feature1), feature3], dim=1)
        final_feature2 = self.feature_final_conv2(up_sample2)
        # 128x128x128
        up_sample3 = torch.cat([self.up_sample1(final_feature2), feature4], dim=1)
        final_feature3 = self.feature_final_conv3(up_sample3)




        ##Feature Pyramid
        class_feature = self.class_heatmap(final_feature3)
        size_map = self.size_map(final_feature3)
        offset_map = self.offset_map(final_feature3)

        return classificaiton, F.sigmoid(class_feature), class_feature, size_map, offset_map