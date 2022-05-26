import torch
import torch.nn.functional as F
from util.helper import CSPInvertedBottleNect


class ShuffleNet(torch.nn.Module):

    def __init__(self, class_num=2, activation=torch.nn.Hardswish):
        super(ShuffleNet, self).__init__()

        self.class_num = class_num

        self.features = torch.nn.Sequential(

        )

        # module 초기화
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view([-1, self.class_num])
        x = torch.softmax(x, dim=1)
        return x