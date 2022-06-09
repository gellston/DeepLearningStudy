import torch
import torch.nn.functional as F

from model.CSPResnet18 import CSPResnet18


class CSPResnet18CenterNet(torch.nn.Module):

    def __init__(self,
                 backbone = CSPResnet18(class_num=257, activation=torch.nn.SiLU),
                 activation=torch.nn.SiLU,
                 pretrained=True):

        super(CSPResnet18CenterNet, self).__init__()

        self.backbone = backbone

        if pretrained == True:
            for param in backbone.parameters():
                param.requires_grad = False

        else:
            for param in backbone.parameters():
                param.requires_grad = True


        ##Feature Pyramid Network
        self.feature_extraction1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=256,
                                                                       out_channels=24,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(24),
                                                       activation())  # 16x16

        self.feature_extraction2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=128,
                                                                       out_channels=24,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(24),
                                                       activation()) #32x32

        self.feature_extraction3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64,
                                                                       out_channels=24,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(24),
                                                       activation()) #64x64

        self.feature_extraction4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=64,
                                                                       out_channels=24,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(24),
                                                       activation()) #128x128


        self.up_sample1 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_sample2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_sample3 = torch.nn.UpsamplingBilinear2d(scale_factor=2)


        self.feature_final_conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
                                                                       out_channels=24,
                                                                       kernel_size=3,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(24),
                                                       activation()) #16x16


        self.feature_final_conv2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
                                                                       out_channels=24,
                                                                       kernel_size=3,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(24),
                                                       activation()) #32x32

        self.feature_final_conv3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
                                                                       out_channels=24,
                                                                       kernel_size=3,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(24),
                                                       activation()) #64x64
        ##Feature Pyramid Network


        ##PredictionMap
        self.class_heatmap = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
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

        self.size_map = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
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

        self.offset_map = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
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
        #for m in self.modules():
        #    if isinstance(m, torch.nn.Conv2d):
        #        torch.nn.init.xavier_uniform_(m.weight)
        #    elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param이랑 scaling param 초기화(?)
        #        m.weight.data.fill_(1)  #
        #        m.bias.data.zero_()



    def forward(self, x):

        ##Feature Extraction
        conv1 = self.backbone.conv1(x) #128x128x64,
        conv2 = self.backbone.conv2(conv1) #64x64x64,
        conv3 = self.backbone.conv3(conv2) #32x32x128,
        conv4 = self.backbone.conv4(conv3) #16x16x256,
        ##Feature Extraction


        ##Feature Pyramid

        feature1 = self.feature_extraction1(conv4)      #16x16x128
        feature2 = self.feature_extraction2(conv3)      #32x32x128
        feature3 = self.feature_extraction3(conv2)      #64x64x128
        feature4 = self.feature_extraction4(conv1)      #128x128x128

        # 32x32x128
        up_sample1 = self.up_sample1(feature1) + feature2
        final_feature1 = self.feature_final_conv1(up_sample1)
        # 64x64x128
        up_sample2 = self.up_sample1(final_feature1) + feature3
        final_feature2 = self.feature_final_conv2(up_sample2)
        # 128x128x128
        up_sample3 = self.up_sample1(final_feature2) + feature4
        final_feature3 = self.feature_final_conv3(up_sample3)

        ##Feature Pyramid
        class_feature = self.class_heatmap(final_feature3)
        size_map = self.size_map(final_feature3)
        offset_map = self.offset_map(final_feature3)

        return torch.sigmoid(class_feature), class_feature, size_map, offset_map