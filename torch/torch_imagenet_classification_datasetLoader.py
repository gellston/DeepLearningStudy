import torch
import torch.nn as nn
import random
import torchvision

from ptflops import get_model_complexity_info
from torchsummary import summary
from torch.utils.data import DataLoader

from model.Resnet50 import Resnet50



USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


## Hyper parameter
training_epochs = 100
batch_size = 20

transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.ToTensor()
            ])

classificationDataset = torchvision.datasets.ImageNet(root="D://학습이미지//imagenet//",
                                                      transform=transform)

# dataset loader
data_loader = DataLoader(dataset=classificationDataset,
                         batch_size=batch_size,  # 배치 크기는 100
                         shuffle=True,
                         drop_last=True)


for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    avg_acc = 0
    total_batch = len(data_loader)
    print('total_batch = ', total_batch)
    for X, Y in data_loader:
        print('x = ', X)
        print('x = ', Y)





print('Learning finished')