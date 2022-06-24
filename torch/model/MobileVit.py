import torch
import torch.nn.functional as F
from util.helper import SeparableActivationConv2d


class MobileVit(torch.nn.Module):

    def __init__(self, class_num=5, activation=torch.nn.ReLU):
        super(MobileVit, self).__init__()

        

    def forward(self, x):

        return x