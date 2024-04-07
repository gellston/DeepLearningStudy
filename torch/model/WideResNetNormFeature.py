import torch
import torch.nn.functional as F
from model.WideResNet import WideResNet

class WideResNetNormFeature(torch.nn.Module):

    def __init__(self, wide_resnet=WideResNet(), target_dim=384) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.stem = wide_resnet.stem
        self.layer1 = wide_resnet.layer1
        self.layer2 = wide_resnet.layer2
        self.layer3 = wide_resnet.layer3

    def project(self, layer2, layer3):
        b,c,h,w = layer3.shape
        layer2 = F.interpolate(layer2, size=(h,w), mode="bilinear", align_corners=False)
        features = torch.cat([layer2, layer3], dim=1)
        b, c, h, w = features.shape
        features = features.reshape(b, c, h * w)
        features = features.transpose(1, 2)
        target_features = F.adaptive_avg_pool1d(features, self.target_dim)
        target_features = target_features.transpose(1, 2)
        target_features = target_features.reshape(b, self.target_dim, h, w)
        return target_features

    def forward(self, x):
        stem = self.stem(x)
        layer1 = self.layer1(stem)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        y = self.project(layer2, layer3)

        return y
