import torch
import torch.nn as nn
import random


from ptflops import get_model_complexity_info
from torchsummary import summary
from torch.utils.data import DataLoader

from model.MobileVit import MobileVit
from util.FIATClassificationDataset import FIATClassificationDataset


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


## Hyper parameter
training_epochs = 30
batch_size = 37
target_accuracy = 1
learning_rate = 0.003
accuracy_threshold = 0.5
## Hyper parameter


model = MobileVit(class_num=4).to(device)
print('==== model info ====')
summary(model, (3, 224, 224))
print('====================')


trace_input = torch.rand(1, 3, 512, 512).to(device, dtype=torch.float32)
model(trace_input)