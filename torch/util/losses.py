import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class CenterNetLoss(torch.nn.Module):
    def __init__(self, beta, alpha=.25, gamma=2):
        super(CenterNetLoss, self).__init__()

        self.size_loss = F.huber_loss()
        self.offset_loss = F.huber_loss()
        self.focal_loss = WeightedFocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, prediction, ground_truth):

        return x