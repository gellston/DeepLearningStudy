import torch
from util.helper import CSPSeparableResidualBlock
from util.helper import SeparableConv2d

class CSPCenterNetResnet18(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.SiLU):
        super(CSPCenterNetResnet18, self).__init__()

        self.class_num = class_num

        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3,
                                                         out_channels=16,
                                                         kernel_size=3,
                                                         stride=2,
                                                         padding=1,
                                                         bias=False),
                                         torch.nn.BatchNorm2d(num_features=16),
                                         activation(),
                                         SeparableConv2d(in_channels=16,
                                                         out_channels=27,
                                                         kernel_size=3,
                                                         stride=2,
                                                         padding=1,
                                                         bias=False,
                                                         activation=activation))

        self.conv2 = torch.nn.Sequential(CSPSeparableResidualBlock(in_dim=27, mid_dim=27, out_dim=27, stride=1,
                                                                   activation=activation),
                                         CSPSeparableResidualBlock(in_dim=27, mid_dim=27, out_dim=38, stride=2,
                                                                   activation=activation))

        self.conv3 = torch.nn.Sequential(CSPSeparableResidualBlock(in_dim=38, mid_dim=38, out_dim=38, stride=1,
                                                                   activation=activation),
                                         CSPSeparableResidualBlock(in_dim=38, mid_dim=38, out_dim=50, stride=2,
                                                                   activation=activation))

        self.conv4 = torch.nn.Sequential(CSPSeparableResidualBlock(in_dim=50, mid_dim=50, out_dim=50, stride=1,
                                                                   activation=activation),
                                         CSPSeparableResidualBlock(in_dim=50, mid_dim=50, out_dim=61, stride=2,
                                                                   activation=activation))

        self.conv5 = torch.nn.Sequential(CSPSeparableResidualBlock(in_dim=61, mid_dim=61, out_dim=61, stride=1,
                                                                   activation=activation),
                                         CSPSeparableResidualBlock(in_dim=61, mid_dim=61, out_dim=72, stride=2,
                                                                   activation=activation))

        self.final_conv = SeparableConv2d(in_channels=72,
                                          out_channels=self.class_num,
                                          kernel_size=3,
                                          bias=False,
                                          padding='same',
                                          activation=activation)

        self.bn = torch.nn.BatchNorm2d(num_features=self.class_num)
        self.global_average_pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.sigmoid = torch.nn.Sigmoid()



        ##Feature Pyramid Network
        self.feature_extraction2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=50,
                                                                       out_channels=60,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(60),
                                                       activation()) #28x28

        self.feature_extraction3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=38,
                                                                       out_channels=60,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(60),
                                                       activation()) #56x56

        self.feature_extraction4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=27,
                                                                       out_channels=60,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(60),
                                                       activation()) #112x112

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=61,
                                                out_channels=60,
                                                kernel_size=2,
                                                bias=False,
                                                stride=2) #28x28

        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=60,
                                                out_channels=60,
                                                kernel_size=2,
                                                bias=False,
                                                stride=2) #56x56

        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=60,
                                                out_channels=60,
                                                kernel_size=2,
                                                bias=False,
                                                stride=2) #112x112



        self.feature_final_conv2 = torch.nn.Sequential(SeparableConv2d(in_channels=60,
                                                                       out_channels=60,
                                                                       kernel_size=3,
                                                                       bias=False,
                                                                       padding='same',
                                                                       activation=activation)) #28x28

        self.feature_final_conv3 = torch.nn.Sequential(SeparableConv2d(in_channels=60,
                                                                       out_channels=60,
                                                                       kernel_size=3,
                                                                       bias=False,
                                                                       padding='same',
                                                                       activation=activation)) #56x56

        self.feature_final_conv4 = torch.nn.Sequential(SeparableConv2d(in_channels=60,
                                                                       out_channels=60,
                                                                       kernel_size=3,
                                                                       bias=False,
                                                                       padding='same',
                                                                       activation=activation)) #112x112

        ##Feature Pyramid Network


        ##PredictionMap
        self.class_heatmap = torch.nn.Sequential(SeparableConv2d(in_channels=60,
                                                                 out_channels=50,
                                                                 kernel_size=3,
                                                                 bias=False,
                                                                 padding='same',
                                                                 activation=activation),
                                                 torch.nn.Conv2d(in_channels=50,
                                                                 out_channels=1,
                                                                 kernel_size=1,
                                                                 bias=True,
                                                                 padding='same'),
                                                 torch.nn.Sigmoid())

        self.size_map = torch.nn.Sequential(SeparableConv2d(in_channels=60,
                                                            out_channels=50,
                                                            kernel_size=3,
                                                            bias=False,
                                                            padding='same',
                                                            activation=activation),
                                            torch.nn.Conv2d(in_channels=50,
                                                            out_channels=2,
                                                            kernel_size=1,
                                                            bias=True,
                                                            padding='same'),
                                            torch.nn.ReLU())

        self.offset_map = torch.nn.Sequential(SeparableConv2d(in_channels=60,
                                                              out_channels=50,
                                                              kernel_size=3,
                                                              bias=False,
                                                              padding='same',
                                                              activation=activation),
                                              torch.nn.Conv2d(in_channels=50,
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

        x = self.final_conv(conv5)
        x = self.bn(x)
        x = self.global_average_pooling(x)
        x = x.view([-1, self.class_num])
        classificaiton = self.sigmoid(x)
        ##Classificaiton


        ##Feature Pyramid
        feature2 = self.feature_extraction2(conv3) #28x28x128
        feature3 = self.feature_extraction3(conv2) #56x56x128
        feature4 = self.feature_extraction4(conv1) #112x112x128

        deconv1 = self.deconv1(conv4) #28x28x128
        final_feature1 = self.feature_final_conv2(torch.add(deconv1, feature2))

        deconv2 = self.deconv2(final_feature1) #56x56x128
        final_feature2 = self.feature_final_conv3(torch.add(deconv2, feature3))

        deconv3 = self.deconv3(final_feature2) #112x112x128
        final_feature3 = self.feature_final_conv4(torch.add(deconv3, feature4))
        ##Feature Pyramid

        class_heatmap = self.class_heatmap(final_feature3)
        size_map = self.size_map(final_feature3)
        offset_map = self.offset_map(final_feature3)

        return classificaiton, class_heatmap, size_map, offset_map