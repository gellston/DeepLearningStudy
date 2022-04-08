import torch
import numpy as np
import torch.nn.functional as F

def IOU(target, prediction):
    prediction = np.where(prediction > 0.5, 1, 0)
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score




class ResidualBlock(torch.nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim, stride=1):
        super(ResidualBlock, self).__init__()

        self.stride = stride;
        self.residual_block = torch.nn.Sequential(torch.nn.Conv2d(in_dim,
                                                                  mid_dim,
                                                                  kernel_size=3,
                                                                  stride=self.stride,
                                                                  padding=1,
                                                                  bias=False),
                                                  torch.nn.BatchNorm2d(num_features=mid_dim),
                                                  torch.nn.SiLU(),
                                                  torch.nn.Conv2d(mid_dim,
                                                                  out_dim,
                                                                  kernel_size=3,
                                                                  padding='same',
                                                                  bias=False),
                                                  torch.nn.BatchNorm2d(num_features=out_dim))

        self.projection = torch.nn.Conv2d(in_channels=in_dim,
                                          out_channels=out_dim,
                                          kernel_size=1,
                                          stride=2)
        self.relu = torch.nn.SiLU()



    def forward(self, x):
        out = self.residual_block(x)  # F(x)
        if self.stride == 2:
            out = torch.add(out, self.projection(x))
        else:
            out = torch.add(out, x)
        out = self.relu(out)

        return out