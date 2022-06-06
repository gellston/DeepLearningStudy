import torch.nn as nn
import torch.nn.functional as F

class PerceptronSimpleMnist(nn.Module):

    def __init__(self):
        super(PerceptronSimpleMnist, self).__init__()
        self.hidden1 = nn.Linear(784, 128) #--> 784 *100 = 78400 W , 28x28
        self.drop_out1 = nn.Dropout(p=0.3)
        self.hidden2 = nn.Linear(128, 64)
        self.drop_out2 = nn.Dropout(p=0.3)
        self.hidden3 = nn.Linear(64, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.drop_out1(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.drop_out2(x)
        x = self.hidden3(x)
        x = self.softmax(x) ## 데이터셋의 이미지상에 객체가 1개가 나오는걸 보장
        return x
