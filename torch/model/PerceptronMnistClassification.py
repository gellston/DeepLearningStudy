#import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F

class PerceptronMnistClassification(nn.Module):

    def __init__(self):
        super(PerceptronMnistClassification, self).__init__()
        self.fc1 = nn.Linear(784, 100) #--> 784 *100 = 78400 W 썻음
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)


        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x
