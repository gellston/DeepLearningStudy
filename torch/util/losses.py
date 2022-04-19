import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):

    def __init__(self, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32).cuda()
        self.eps = 1e-6

    def forward(self, input, target):
        # input are not the probabilities, they are just the cnn out vector
        # input and target shape: (bs, n_classes)
        # sigmoid

        log_probs = -torch.log(input)

        focal_loss = torch.sum(torch.pow(1 - input + self.eps, self.gamma).mul(log_probs).mul(target), dim=1)
        # bce_loss = torch.sum(log_probs.mul(target), dim = 1)

        return focal_loss.mean()  # , bce_loss

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma
        self.bce_loss = torch.nn.BCELoss()


    def forward(self, inputs, targets):
        BCE_loss = self.bce_loss(inputs, targets)   #CrossEntropy
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class CenterNetLoss(torch.nn.Module):
    def __init__(self, alpha=.25, gamma=2, lambda_size=0.1, lambda_offset=1):
        super(CenterNetLoss, self).__init__()

        self.lambda_size = lambda_size
        self.lambda_offset = lambda_offset

        self.focal_loss = WeightedFocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, prediction_heatmap,
                      prediction_sizemap,
                      prediction_offsetmap,
                      label_heatmap,
                      label_sizemap,
                      label_offsetmap):

        class_loss = self.focal_loss(prediction_heatmap, label_heatmap)
        size_loss = F.huber_loss(prediction_sizemap, label_sizemap, reduction='sum')
        offset_loss = F.huber_loss(prediction_offsetmap, label_offsetmap, reduction='sum')

        return class_loss + size_loss * self.lambda_size + offset_loss * self.lambda_offset;