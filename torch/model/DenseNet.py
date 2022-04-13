import torch
import math
from util.helper import DenseBlock
from util.helper import Transition

class DenseNet(torch.nn.Module):

    def __init__(self, class_num=5, block_config=(6, 12, 24, 16), expansion_rate=4, growth_rate=12, droprate=0.2):

        super(DenseNet, self).__init__()

        self.class_num = class_num
        self.block_config = block_config                ## filter크기 배열
        self.growth_rate = growth_rate
        self.expansion_rate = expansion_rate


        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=growth_rate * 2,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False)
        )

        self.features = torch.nn.Sequential()

        inner_channels = growth_rate * 2
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_input_features=inner_channels,
                               num_layers=num_layers,
                               expansion_rate=self.expansion_rate,
                               growth_rate=self.growth_rate,
                               droprate=droprate)

            self.features.add_module("denseblock_%d" % (i+1), block)
            inner_channels = inner_channels + num_layers * growth_rate

            if i != len(block_config) - 1:
                transition = Transition(in_channels=inner_channels,
                                        out_channels=int(inner_channels / 2),
                                        droprate=droprate)
                self.features.add_module("transition_%d" % (i + 1), transition)
                inner_channels = int(inner_channels / 2)

        self.features.add_module("final_norm", torch.nn.BatchNorm2d(inner_channels))
        self.features.add_module("final_relu", torch.nn.ReLU())
        self.class_conv = torch.nn.Conv2d(in_channels=inner_channels,
                                          out_channels=self.class_num,
                                          kernel_size=3,
                                          bias=False,
                                          padding='same')
        self.final_batch_norm = torch.nn.BatchNorm2d(self.class_num)
        self.global_average_pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.softmax = torch.nn.Softmax(dim=1)

        # module 초기화
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):  # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1)  #
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.class_conv(x)
        x = self.final_batch_norm(x)
        x = self.global_average_pooling(x)
        x = x.view([-1, self.class_num])
        x = self.softmax(x)

        return x