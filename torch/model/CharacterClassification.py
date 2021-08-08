#import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F

class CharacterClassification(nn.Module):

    def __init__(self):
        super(CharacterClassification, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.float()
        h1 = torch.relu(self.fc1(x.view(-1, 784)))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.relu(self.fc3(h2))
        h4 = torch.relu(self.fc4(h3))
        h5 = torch.relu(self.fc5(h4))
        h6 = self.fc6(h5)
        return torch.softmax(h6, dim=1)
