import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision







class CenterNetLoss(torch.nn.Module):
    def __init__(self, alpha=.25, gamma=2, lambda_size=0.1, lambda_offset=1):
        super(CenterNetLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.lambda_size = lambda_size
        self.lambda_offset = lambda_offset

    def forward(self, prediction_features,  ##sigmoid focal loss함수에서 sigmoid를 쒸우기때문에 sigmoid없는 feature map 입력
                      prediction_sizemap,
                      prediction_offsetmap,
                      label_heatmap,
                      label_sizemap,
                      label_offsetmap):


        class_loss = torchvision.ops.sigmoid_focal_loss(prediction_features, label_heatmap, self.alpha, self.gamma, reduction='mean')
        size_loss = F.huber_loss(prediction_sizemap, label_sizemap, reduction='mean')
        offset_loss = F.huber_loss(prediction_offsetmap, label_offsetmap, reduction='mean')

        return class_loss #+ size_loss * self.lambda_size + offset_loss * self.lambda_offset;