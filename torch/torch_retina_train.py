import torch
import random

from torch.utils.data import DataLoader

from util.ODDatasetLoader import ODDatasetLoader


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


training_epochs = 300
batch_size = 1
target_accuracy = 0.99


datasets = ODDatasetLoader('C://Users//wantr//Desktop//LabelTest',
                           label_height=256,
                           label_width=256,
                           isColor=True,
                           isNorm=False)

data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)



for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0

    for X, Y in data_loader:
        print("check")


