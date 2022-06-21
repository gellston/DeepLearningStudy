import torch
import torch.nn.functional as F
import torch.nn as nn


class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        ##28x28
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5,
                               stride=1)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2,
                                      stride=2)


        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               stride=1)

        self.avg_pool2 = nn.AvgPool2d(kernel_size=2,
                                      stride=2)

        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=120,
                               kernel_size=4,
                               stride=1)


        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=120,
                             out_features=84)

        self.fc2 = nn.Linear(in_features=84,
                             out_features=10)


    def forward(self, x):

        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.avg_pool1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.avg_pool2(x)
        x = torch.tanh(x) #--->4x4x?
        x = self.conv3(x) ## --->4x4 ------> 1x1xfilter


        x = self.flatten(x)
        x = torch.tanh(x)

        x = self.fc1(x)
        x = torch.tanh(x)

        x = self.fc2(x)

        x = F.softmax(x, dim=1)



        return x