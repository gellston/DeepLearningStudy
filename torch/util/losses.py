import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def neg_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt   (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss



class CenterNetFocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super(CenterNetFocalLoss, self).__init__()
        self.neg_loss = neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def jaccard_loss(logits, true,  eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)




class CenterNetLoss(torch.nn.Module):
    def __init__(self, alpha=.25, gamma=2, lambda_size=0.1, lambda_offset=1):
        super(CenterNetLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.lambda_size = lambda_size
        self.lambda_offset = lambda_offset

        self.size_loss = torch.nn.SmoothL1Loss(reduction='sum')
        self.offset_loss = torch.nn.SmoothL1Loss(reduction='sum')
        self.focal_loss = CenterNetFocalLoss()
        self.iou_loss_function = nn.SmoothL1Loss(reduction='none')


    def forward(self, prediction_features,  ##sigmoid focal loss함수에서 sigmoid를 쒸우기때문에 sigmoid없는 feature map 입력
                      prediction_sizemap,
                      prediction_offsetmap,
                      label_heatmap,
                      label_sizemap,
                      label_offsetmap,
                      label_bbox_count,
                      batch_size):


        clamped_sigmoid = torch.clamp(torch.sigmoid(prediction_features), min=1e-4, max=1 - 1e-4)
        class_loss = self.focal_loss(clamped_sigmoid, label_heatmap)
        size_loss = self.size_loss(prediction_sizemap, label_sizemap) / label_bbox_count
        offset_loss = self.offset_loss(prediction_offsetmap, label_offsetmap) / label_bbox_count
        iou_loss = torch.sum(self.iou_loss_function(torch.sigmoid(prediction_features), label_heatmap) * label_heatmap) / batch_size

        return class_loss + size_loss * self.lambda_size + offset_loss * self.lambda_offset + iou_loss