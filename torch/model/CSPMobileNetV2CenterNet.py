 import torch
import torch.nn.functional as F

from model.CSPMobileNetV2 import CSPMobileNetV2


class CSPMobileNetV2CenterNet(torch.nn.Module):

    def __init__(self,
                 backbone = CSPMobileNetV2(class_num=257, activation=torch.nn.ReLU6),
                 pretrained=True):

        super(CSPMobileNetV2CenterNet, self).__init__()

        self.backbone = backbone

        if pretrained == True:
            for param in backbone.parameters():
                param.requires_grad = False

        else:
            for param in backbone.parameters():
                param.requires_grad = True

        ##Feature Pyramid Network
        self.feature_extraction1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=320,
                                                                       out_channels=24,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(24, eps=0.001, momentum=0.9),
                                                       torch.nn.ReLU())  # 16x16

        self.feature_extraction2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=96,
                                                                       out_channels=24,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(24, eps=0.001, momentum=0.9),
                                                       torch.nn.ReLU())  # 32x32

        self.feature_extraction3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=32,
                                                                       out_channels=24,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(24, eps=0.001, momentum=0.9),
                                                       torch.nn.ReLU())  # 64x64

        self.feature_extraction4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
                                                                       out_channels=24,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(24, eps=0.001, momentum=0.9),
                                                       torch.nn.ReLU())  # 128x128

        self.up_sample1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=24, out_channels=24, bias=False,
                                                                       kernel_size=2, stride=2),
                                              torch.nn.BatchNorm2d(24, eps=0.001, momentum=0.9),
                                              torch.nn.ReLU())
        self.up_sample2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=24, out_channels=24, bias=False,
                                                                       kernel_size=2, stride=2),
                                              torch.nn.BatchNorm2d(24, eps=0.001, momentum=0.9),
                                              torch.nn.ReLU())
        self.up_sample3 = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=24, out_channels=24, bias=False,
                                                                       kernel_size=2, stride=2),
                                              torch.nn.BatchNorm2d(24, eps=0.001, momentum=0.9),
                                              torch.nn.ReLU())

        self.feature_final = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
                                                                 out_channels=24,
                                                                 kernel_size=3,
                                                                 bias=False,
                                                                 padding='same'),
                                                 torch.nn.BatchNorm2d(24, eps=1e-5, momentum=0.99),
                                                 torch.nn.ReLU())

        ##Feature Pyramid Network

        ##PredictionMap
        self.class_heatmap = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
                                                                 out_channels=1,
                                                                 kernel_size=1,
                                                                 bias=True,
                                                                 padding='same'))

        self.size_map = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
                                                            out_channels=2,
                                                            kernel_size=1,
                                                            bias=True,
                                                            padding='same'))

        self.offset_map = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
                                                              out_channels=2,
                                                              kernel_size=1,
                                                              bias=True,
                                                              padding='same'))
        ##PredictionMap

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

            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):

        ##Feature Extraction
        conv1 = self.backbone.features[0](x)  # 320x320x32,
        conv2 = self.backbone.features[1](conv1)  # 320x320x32,
        conv3 = self.backbone.features[2](conv2)  # 320x320x32,
        conv4 = self.backbone.features[3](conv3)  # 320x320x16
        conv5 = self.backbone.features[4](conv4)  # 160x160x24,
        conv6 = self.backbone.features[5](conv5)  # 160x160x24,
        conv7 = self.backbone.features[6](conv6)  # 80x80x32,
        conv8 = self.backbone.features[7](conv7)  # 80x80x32,
        conv9 = self.backbone.features[8](conv8)  # 80x80x32,
        conv10 = self.backbone.features[9](conv9)  # 40x40x64,
        conv11 = self.backbone.features[10](conv10)  # 40x40x64,
        conv12 = self.backbone.features[11](conv11)  # 40x40x64,
        conv13 = self.backbone.features[12](conv12)  # 40x40x64,
        conv14 = self.backbone.features[13](conv13)  # 40x40x64,
        conv15 = self.backbone.features[14](conv14)  # 40x40x64,
        conv16 = self.backbone.features[15](conv15)  # 40x40x96,
        conv17 = self.backbone.features[16](conv16)  # 20x20x160
        conv18 = self.backbone.features[17](conv17)  # 20x20x160
        conv19 = self.backbone.features[18](conv18)  # 20x20x160
        conv20 = self.backbone.features[19](conv19)  # 20x20x320
        ##Feature Extraction


        ##Feature Pyramid
        feature1 = self.feature_extraction1(conv20)  # 20x20x320  -> 24
        feature2 = self.feature_extraction2(conv14)  # 20x20x160  -> 24
        feature3 = self.feature_extraction3(conv7)  # 40x40x64   -> 24
        feature4 = self.feature_extraction4(conv5)  # 80x80x24    -> 24


        up_sample1 = self.up_sample1(feature1) + feature2
        up_sample2 = self.up_sample2(up_sample1) + feature3
        up_sample3 = self.up_sample3(up_sample2) + feature4
        final_feature3 = self.feature_final(up_sample3)
        ##Feature Pyramid

        ##Feature Head
        class_feature = self.class_heatmap(final_feature3)
        size_map = self.size_map(final_feature3)
        offset_map = self.offset_map(final_feature3)
        ##Feature Head

        return torch.sigmoid(class_feature), class_feature, size_map, offset_map