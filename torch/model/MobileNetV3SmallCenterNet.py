import torch

from model.MobileNetV3Small import MobileNetV3Small


class MobileNetV3SmallCenterNet(torch.nn.Module):

    def __init__(self,
                 fpn_conv_filters=64):
        super(MobileNetV3SmallCenterNet, self).__init__()

        self.fpn_conv_filters = fpn_conv_filters
        self.backbone = MobileNetV3Small(class_num=1)

        ##Feature Pyramid Network
        self.feature_extraction1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=96,
                                                                       out_channels=self.fpn_conv_filters,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(self.fpn_conv_filters, eps=0.001, momentum=0.9),
                                                       torch.nn.ReLU())  # 16x16

        self.feature_extraction2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=48,
                                                                       out_channels=self.fpn_conv_filters,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(self.fpn_conv_filters, eps=0.001, momentum=0.9),
                                                       torch.nn.ReLU())  # 32x32

        self.feature_extraction3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=24,
                                                                       out_channels=self.fpn_conv_filters,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(self.fpn_conv_filters, eps=0.001, momentum=0.9),
                                                       torch.nn.ReLU())  # 64x64

        self.feature_extraction4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=16,
                                                                       out_channels=self.fpn_conv_filters,
                                                                       kernel_size=1,
                                                                       bias=False,
                                                                       padding='same'),
                                                       torch.nn.BatchNorm2d(self.fpn_conv_filters, eps=0.001, momentum=0.9),
                                                       torch.nn.ReLU())  # 128x128

        self.up_sample1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=self.fpn_conv_filters, out_channels=self.fpn_conv_filters, bias=False,
                                                                       kernel_size=2, stride=2),
                                              torch.nn.BatchNorm2d(self.fpn_conv_filters, eps=0.001, momentum=0.9),
                                              torch.nn.ReLU())
        self.up_sample2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=self.fpn_conv_filters, out_channels=self.fpn_conv_filters, bias=False,
                                                                       kernel_size=2, stride=2),
                                              torch.nn.BatchNorm2d(self.fpn_conv_filters, eps=0.001, momentum=0.9),
                                              torch.nn.ReLU())
        self.up_sample3 = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels=self.fpn_conv_filters, out_channels=self.fpn_conv_filters, bias=False,
                                                                       kernel_size=2, stride=2),
                                              torch.nn.BatchNorm2d(self.fpn_conv_filters, eps=0.001, momentum=0.9),
                                              torch.nn.ReLU())

        self.feature_final = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.fpn_conv_filters,
                                                                 out_channels=self.fpn_conv_filters,
                                                                 kernel_size=3,
                                                                 bias=False,
                                                                 padding='same'),
                                                 torch.nn.BatchNorm2d(self.fpn_conv_filters, eps=1e-5, momentum=0.99),
                                                 torch.nn.ReLU())

        ##Feature Pyramid Network

        ##PredictionMap
        self.class_heatmap = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.fpn_conv_filters,
                                                                 out_channels=1,
                                                                 kernel_size=1,
                                                                 bias=True,
                                                                 padding='same'))

        self.size_map = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.fpn_conv_filters,
                                                            out_channels=2,
                                                            kernel_size=1,
                                                            bias=True,
                                                            padding='same'))

        self.offset_map = torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.fpn_conv_filters,
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

    def train_mode(self):
        self.backbone.train()
        self.train()

    def eval_mode(self):
        self.backbone.eval()
        self.eval()

    def forward(self, x):

        ##Feature Extraction
        conv1 = self.backbone.features[0](x)
        conv2 = self.backbone.features[1](conv1)
        conv3 = self.backbone.features[2](conv2)
        conv4 = self.backbone.features[3](conv3)
        conv5 = self.backbone.features[4](conv4)
        conv6 = self.backbone.features[5](conv5)
        conv7 = self.backbone.features[6](conv6)
        conv8 = self.backbone.features[7](conv7)
        conv9 = self.backbone.features[8](conv8)
        conv10 = self.backbone.features[9](conv9)
        conv11 = self.backbone.features[10](conv10)
        conv12 = self.backbone.features[11](conv11)
        conv13 = self.backbone.features[12](conv12)
        conv14 = self.backbone.features[13](conv13)
        ##Feature Extraction




        ##Feature Pyramid
        feature1 = self.feature_extraction1(conv14)
        feature2 = self.feature_extraction2(conv11)
        feature3 = self.feature_extraction3(conv6)
        feature4 = self.feature_extraction4(conv4)


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